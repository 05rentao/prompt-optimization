#!/usr/bin/env python3
"""Adversary-only fine-tuning runner using dataset prompts and model judgment.

This script intentionally excludes prompt-optimization steps (for example GEPA or
reflection loops). It trains only the adversary model weights with REINFORCE, RLOO,
or rejection-sampling policy-gradient updates from target responses judged by
either HarmBench judge verdicts or the shared heuristic evaluator.

Use this entrypoint for Theme 2 sweeps (prompt modes, policy, iterations); see
``notes/theme2-adversary-experiments.md``. For a minimal fixed-prompt baseline,
see ``runs/coev_v2_run.py``. Use ``--no-finetune`` (train mode) to sweep rewriter
prompts without LoRA updates. Dataset sizing: ``hf_train_size`` (HF train pool),
``csv_val_size`` / ``csv_test_size`` + ``harmbench_csv_path`` (periodic val vs
baseline/final test).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pandas as pd
import seaborn as sns
import torch

from src.artifacts import (
    build_baseline_optimized_df,
    log_saved_artifacts,
    save_adversary_asr_vs_iterations_plot,
    save_adversary_refusal_vs_iterations_plot,
    save_baseline_optimized_plot,
    write_json,
    write_many_csv,
    write_run_manifest,
)
from src.data import load_harmbench_csv_val_test_splits, load_harmbench_subset, resolve_harmbench_csv_path
from src.evaluators import compute_refusal_score
from src.run_pipeline import (
    adversary_rewrite_sample,
    adversary_rewriter_system_content,
    build_prompt_pool,
    compute_reward_and_verdict,
    maybe_save_adapters as maybe_save_adapters_common,
)
from src.runtime import (
    EvaluationConfig,
    EvaluationResult,
    GenerationRequest,
    GenerationSession,
    HarmbenchJudgeConfig,
    ModelConfig,
    RuntimeCatalog,
    UnslothAdversaryConfig,
    build_vllm_target_session,
    cap_thread_workers,
    evaluate_outputs,
    patch_run_args_from_config,
    resolve_hf_token,
    timed_target_generate,
)
from src.runtime.adversary_prompts import ADVERSARY_PROMPT_VARIANTS, resolve_adversary_attacker_instruction
from src.runtime.defaults import build_config_snapshot, load_default_config
from src.runtime.policy_gradient import (
    pad_gen_ids_batch,
    reinforce_update_batch_sgd,
    rejection_sampling_update_sgd,
    rloo_update_batch_sgd,
)
from src.types import RunManifest


@dataclass
class AdversaryTrainingConfig:
    """Config values controlling adversary-only fine-tuning and eval cadence.

    Effective training hyperparameters come from YAML (``runs.adversary`` merged with
    ``shared_generation`` via ``load_default_config``); these defaults are only used
    if a code path constructs the dataclass without the merged file.
    """

    iterations: int = 20
    lr: float = 5e-5
    weight_decay: float = 0.01
    eval_every: int = 4
    max_new_tokens: int = 120
    target_max_new_tokens: int = 150
    training_csv_name: str = "adversary_training_log.csv"
    eval_csv_name: str = "adversary_eval_outputs.csv"


@dataclass
class RunContext:
    """Long-lived runtime sessions and device state for one script run."""

    adversary_session: GenerationSession
    target_session: GenerationSession
    judge_session: GenerationSession
    device: str


def clean_text(value: Any) -> str:
    """Normalize values for single-line artifact rows."""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def adversary_output(
    prompt: str,
    attacker_instruction: str,
    adversary_session: GenerationSession,
    device: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any], float]:
    """Generate one rewritten attack prompt from the adversary policy; return elapsed ms."""
    return adversary_rewrite_sample(
        adversary_session,
        device,
        prompt,
        attacker_instruction,
        max_new_tokens=max_new_tokens,
    )


def target_generate(
    prompt: str,
    target_session: GenerationSession,
    device: str,
    target_system_prompt: str,
    max_new_tokens: int = 150,
) -> tuple[str, float]:
    """Generate one target response for a rewritten prompt; return elapsed ms."""

    request = GenerationRequest(
        system_prompt=target_system_prompt,
        user_prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    return timed_target_generate(target_session, device, request)


def reward_from_response(
    prompt: str,
    target_response: str,
    eval_cfg: EvaluationConfig,
    ctx: RunContext,
) -> tuple[float, str]:
    """Compute reward and verdict label from target response quality."""
    return compute_reward_and_verdict(
        behavior=prompt,
        generation=target_response,
        eval_cfg=eval_cfg,
        judge_session=ctx.judge_session if eval_cfg.method == "judge" else None,
    )


def _evaluation_result_to_metrics(ev: EvaluationResult, latency_ms_mean: float) -> dict[str, float]:
    """Align aggregate metric keys with CoEV / ``evaluate_prompts`` JSON exports."""
    aggregate_score = ev.mean_refusal_score
    if aggregate_score is None:
        aggregate_score = ev.refusal_rate
    out: dict[str, float] = {
        "refusal_rate": float(ev.refusal_rate),
        "asr": float(ev.asr),
        "aggregate_score": float(aggregate_score),
        "latency_ms_mean": float(latency_ms_mean),
    }
    if ev.mean_refusal_score is not None:
        out["mean_refusal_score"] = float(ev.mean_refusal_score)
    return out


def evaluate_asr(
    eval_examples: list[dict[str, Any]],
    attacker_instruction: str,
    target_system_prompt: str,
    train_cfg: AdversaryTrainingConfig,
    eval_cfg: EvaluationConfig,
    ctx: RunContext,
    target_max_workers: int | None = None,
) -> tuple[EvaluationResult, list[dict[str, Any]]]:
    """Run held-out eval prompts and return aggregate metrics plus row payloads.

    Per example the chain is adversary → target, but across examples we **pipeline**:
    after each adversary finishes we submit that target request to a pool while the main
    thread runs the next adversary (Unsloth stays sequential; vLLM can overlap in flight).
    """
    n = len(eval_examples)
    workers = cap_thread_workers(n, target_max_workers)

    use_target_pool = getattr(ctx.target_session.runtime, "supports_concurrent_target_inference", False)

    adv_rows: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    target_texts: list[str] = []
    target_latencies: list[float] = []

    if not use_target_pool:
        for i, ex in enumerate(eval_examples, start=1):
            print(f"evaluate_asr {i}/{n}", flush=True)
            sample, adv_latency_ms = adversary_output(
                prompt=ex["prompt"],
                attacker_instruction=attacker_instruction,
                adversary_session=ctx.adversary_session,
                device=ctx.device,
                max_new_tokens=train_cfg.max_new_tokens,
            )
            request = GenerationRequest(
                system_prompt=target_system_prompt,
                user_prompt=sample["completion_text"].strip(),
                max_new_tokens=train_cfg.target_max_new_tokens,
                temperature=0.0,
            )
            target_resp, target_latency_ms = timed_target_generate(ctx.target_session, ctx.device, request)
            adv_rows.append((ex, sample, adv_latency_ms))
            target_texts.append(target_resp)
            target_latencies.append(target_latency_ms)
    else:
        pending: list[tuple[dict[str, Any], dict[str, Any], float, Future[tuple[str, float]]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i, ex in enumerate(eval_examples, start=1):
                print(f"evaluate_asr {i}/{n}", flush=True)
                sample, adv_latency_ms = adversary_output(
                    prompt=ex["prompt"],
                    attacker_instruction=attacker_instruction,
                    adversary_session=ctx.adversary_session,
                    device=ctx.device,
                    max_new_tokens=train_cfg.max_new_tokens,
                )
                request = GenerationRequest(
                    system_prompt=target_system_prompt,
                    user_prompt=sample["completion_text"].strip(),
                    max_new_tokens=train_cfg.target_max_new_tokens,
                    temperature=0.0,
                )
                fut = executor.submit(timed_target_generate, ctx.target_session, ctx.device, request)
                pending.append((ex, sample, adv_latency_ms, fut))

            for ex, sample, adv_latency_ms, fut in pending:
                adv_rows.append((ex, sample, adv_latency_ms))
                target_resp, target_latency_ms = fut.result()
                target_texts.append(target_resp)
                target_latencies.append(target_latency_ms)

    behaviors: list[str] = []
    generations: list[str] = []
    rows: list[dict[str, Any]] = []
    for (ex, sample, adv_latency_ms), target_response, target_latency_ms in zip(
        adv_rows, target_texts, target_latencies, strict=True
    ):
        prompt = ex["prompt"]
        adversary_text = sample["completion_text"].strip()
        refusal_score = compute_refusal_score(target_response)
        total_latency = adv_latency_ms + target_latency_ms
        behaviors.append(prompt)
        generations.append(target_response)
        rows.append(
            {
                "id": str(ex.get("id", "")),
                "behavior": prompt,
                "generation": target_response,
                "prompt": clean_text(prompt),
                "adversary_prompt": clean_text(adversary_text),
                "target_response": clean_text(target_response),
                "refusal_score": refusal_score,
                "asr_score": 1.0 - refusal_score,
                "adversary_latency_ms": adv_latency_ms,
                "target_latency_ms": target_latency_ms,
                "latency_ms_total": total_latency,
                "latency_ms": total_latency,
            }
        )

    eval_result = evaluate_outputs(
        behaviors=behaviors,
        generations=generations,
        cfg=eval_cfg,
        judge_session=ctx.judge_session if eval_cfg.method == "judge" else None,
    )
    return eval_result, rows


def _eval_suite(
    eval_examples: list[dict[str, Any]],
    attacker_instruction: str,
    target_system_prompt: str,
    train_cfg: AdversaryTrainingConfig,
    eval_cfg: EvaluationConfig,
    ctx: RunContext,
    args: argparse.Namespace,
) -> tuple[EvaluationResult, list[dict[str, Any]], dict[str, float]]:
    """Baseline / periodic / final eval with shared kwargs."""
    ev, rows = evaluate_asr(
        eval_examples=eval_examples,
        attacker_instruction=attacker_instruction,
        target_system_prompt=target_system_prompt,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        ctx=ctx,
        target_max_workers=args.target_max_workers,
    )
    df = pd.DataFrame(rows)
    lat_mean = float(df["latency_ms_total"].mean()) if not df.empty else 0.0
    metrics = _evaluation_result_to_metrics(ev, lat_mean)
    return ev, rows, metrics


def _try_save_adapters_print(ctx: RunContext, save_dir: str | None, results_dir: Path) -> str | None:
    path = maybe_save_adapters_common(ctx.adversary_session, save_dir, results_dir=results_dir)
    if path:
        print(f"Saved adversary adapters to: {path}")
    return path


def _adversary_training_label(args: argparse.Namespace) -> str:
    """Human-readable label for plot copy (REINFORCE, RLOO, or rejection sampling)."""

    if not getattr(args, "finetune", True):
        return "no finetune (prompt eval)"
    if getattr(args, "rs_min_successes", 0) > 0:
        return "rejection sampling"
    if getattr(args, "adversary_policy", "reinforce") == "rloo":
        return "RLOO"
    return "REINFORCE"


def save_artifacts(
    args: argparse.Namespace,
    results_dir: Path,
    train_cfg: AdversaryTrainingConfig,
    baseline_metrics: dict[str, float],
    final_metrics: dict[str, float],
    baseline_df: pd.DataFrame,
    final_df: pd.DataFrame,
    training_df: pd.DataFrame,
    run_seconds: float,
    adapter_path: str | None,
    config_snapshot: dict[str, Any],
) -> None:
    """Persist metrics, CSVs, plots, and manifest.

    Prompts are fixed for the run (from CLI/config); artifacts compare **adversary
    weights** before vs after policy-gradient training, not prompt optimization.
    Full rewriter and target system prompt strings are recorded in ``run_manifest.json``
    under ``extra.prompts``.

    When there is no fine-tuning (``--no-finetune`` or ``--mode eval``), only one test
    eval exists; we write ``eval_outputs.csv`` / ``eval_metrics.csv`` and omit the
    before/after comparison plot and duplicate CSVs.

    Prompts are written to ``adversary_run_metrics.json`` (``prompts``), ``prompts.json``,
    and ``run_manifest.json`` (``extra.prompts``).
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    compare_before_after = args.mode == "train" and getattr(args, "finetune", True)

    config_block: dict[str, Any] = {
        "mode": args.mode,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "hf_train_size": args.hf_train_size,
        "csv_val_size": args.csv_val_size,
        "csv_test_size": args.csv_test_size,
        "harmbench_csv_path": getattr(args, "harmbench_csv_resolved_path", None),
        "csv_seed": args.csv_seed,
        "runtime_profile": args.runtime_profile,
        "target_model_name": args.task_model_name,
        "reflection_vllm_base_url": args.reflection_vllm_base_url,
        "eval_method": args.eval_method,
        "iterations": (
            train_cfg.iterations
            if args.mode == "train" and getattr(args, "finetune", True)
            else 0
        ),
        "finetune": getattr(args, "finetune", True),
        "run_seconds": run_seconds,
        "adversary_policy": getattr(args, "adversary_policy", "reinforce"),
        "adversary_prompt": getattr(args, "adversary_prompt", "default"),
        "rs_budget": getattr(args, "rs_budget", 5),
        "rs_min_successes": getattr(args, "rs_min_successes", 0),
    }
    prompts_payload = {
        "adversary_prompt_variant": getattr(args, "adversary_prompt", "default"),
        "adversary_attacker_instruction": args.attacker_instruction,
        "adversary_rewriter_system_message": adversary_rewriter_system_content(
            args.attacker_instruction
        ),
        "target_system_prompt": args.target_system_prompt,
    }
    if compare_before_after:
        metrics_payload: dict[str, Any] = {
            "config": config_block,
            "eval_kind": "before_after_training",
            "prompts": prompts_payload,
            "baseline_metrics": baseline_metrics,
            "final_metrics": final_metrics,
        }
    else:
        metrics_payload = {
            "config": config_block,
            "eval_kind": "single_test_eval",
            "prompts": prompts_payload,
            "test_eval_metrics": baseline_metrics,
        }
    metrics_json_path = results_dir / "adversary_run_metrics.json"
    write_json(metrics_json_path, metrics_payload)
    prompts_json_path = results_dir / "prompts.json"
    write_json(prompts_json_path, prompts_payload)

    csv_files: dict[str, pd.DataFrame] = {}
    if compare_before_after:
        comparison_df = build_baseline_optimized_df(
            baseline_metrics=baseline_metrics,
            optimized_metrics=final_metrics,
            baseline_variant="before_training",
            comparison_variant="after_training",
        )
        csv_files["eval_metrics_before_vs_after_training.csv"] = comparison_df
        csv_files["eval_outputs_before_training.csv"] = baseline_df
        csv_files["eval_outputs_after_training.csv"] = final_df
    else:
        csv_files["eval_metrics.csv"] = pd.DataFrame([{"variant": "test_eval", **baseline_metrics}])
        csv_files["eval_outputs.csv"] = baseline_df
    csv_files[train_cfg.training_csv_name] = training_df
    write_many_csv(
        results_dir,
        csv_files,
        skip_empty={train_cfg.training_csv_name},
    )

    plot_path: Path | None
    if compare_before_after:
        policy_label = _adversary_training_label(args)
        eval_subtitle = (
            "Same fixed rewriter instruction and target system prompt. "
            f"before_training = adversary LoRA at init; after_training = same LoRA after {policy_label}. "
            "Bars = aggregate metrics on the held-out eval set (not per-step noise)."
        )
        plot_path = save_baseline_optimized_plot(
            comparison_df=comparison_df,
            out_path=results_dir / "plot_eval_metrics_before_vs_after_training.png",
            title="Adversary eval: before vs after training",
            subtitle=eval_subtitle,
        )
    else:
        plot_path = None
    train_iters = (
        train_cfg.iterations
        if args.mode == "train" and getattr(args, "finetune", True)
        else None
    )
    final_asr_train = final_metrics.get("asr") if args.mode == "train" else None
    final_refusal_train = (1.0 - float(final_asr_train)) if final_asr_train is not None else None
    asr_iter_path = save_adversary_asr_vs_iterations_plot(
        training_df=training_df,
        out_path=results_dir / "plot_asr_vs_iterations.png",
        title="Adversary ASR vs training iteration",
        final_asr=final_asr_train,
        iterations=train_iters,
    )
    refusal_iter_path = save_adversary_refusal_vs_iterations_plot(
        training_df=training_df,
        out_path=results_dir / "plot_refusal_vs_iterations.png",
        title="Adversary refusal rate vs training iteration",
        final_refusal=final_refusal_train,
        iterations=train_iters,
    )

    manifest = RunManifest(
        mode=f"adversary_{args.mode}",
        runtime_profile=args.runtime_profile,
        seed=args.seed,
        dataset={
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "hf_train_size": args.hf_train_size,
            "csv_val_size": args.csv_val_size,
            "csv_test_size": args.csv_test_size,
            "harmbench_csv_path": getattr(args, "harmbench_csv_resolved_path", None),
            "csv_seed": args.csv_seed,
        },
        models={
            "adversary_model": args.adversary_model_id,
            "target_model": args.task_model_name,
            "judge_model": args.judge_model_id,
        },
        budget={
            "iterations": (
                train_cfg.iterations
                if args.mode == "train" and getattr(args, "finetune", True)
                else 0
            ),
            "eval_every": train_cfg.eval_every if args.mode == "train" and getattr(args, "finetune", True) else 0,
            "run_seconds": run_seconds,
        },
        endpoints={"reflection_base_url": args.reflection_vllm_base_url},
        extra={
            "metrics_json_path": str(metrics_json_path),
            "prompts_json_path": str(prompts_json_path),
            "eval_artifact_layout": (
                "before_after_training" if compare_before_after else "single_test_eval"
            ),
            "comparison_metrics_csv": (
                "eval_metrics_before_vs_after_training.csv" if compare_before_after else None
            ),
            "eval_metrics_csv": "eval_metrics.csv" if not compare_before_after else None,
            "eval_outputs_csv": (
                ["eval_outputs_before_training.csv", "eval_outputs_after_training.csv"]
                if compare_before_after
                else ["eval_outputs.csv"]
            ),
            "training_csv_name": train_cfg.training_csv_name,
            "eval_csv_name": train_cfg.eval_csv_name,
            "save_dir": adapter_path,
            "eval_method": args.eval_method,
            "adversary_policy": getattr(args, "adversary_policy", "reinforce"),
            "adversary_prompt": getattr(args, "adversary_prompt", "default"),
            "rs_budget": getattr(args, "rs_budget", 5),
            "rs_min_successes": getattr(args, "rs_min_successes", 0),
            "finetune": getattr(args, "finetune", True),
            "prompts": prompts_payload,
        },
        config_snapshot=config_snapshot,
    )
    manifest_path = write_run_manifest(results_dir=results_dir, payload=manifest)
    logged_paths: list[Path] = [metrics_json_path, prompts_json_path, manifest_path]
    if plot_path is not None:
        logged_paths.append(plot_path)
    logged_paths.append(results_dir)
    if asr_iter_path is not None:
        logged_paths.insert(-1, asr_iter_path)
    if refusal_iter_path is not None:
        logged_paths.insert(-1, refusal_iter_path)
    log_saved_artifacts(logged_paths)


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse CLI flags for adversary-only fine-tuning and evaluation."""
    global_defaults = defaults["global"]
    run_defaults = defaults["runs"]["adversary"]
    parser = argparse.ArgumentParser(description="Run adversary-only fine-tuning (no prompt optimization).")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--device", default=global_defaults["device"], help="Device override (e.g. cuda, cpu).")

    parser.add_argument("--dataset-name", default=global_defaults["dataset_name"])
    parser.add_argument("--dataset-config", default=global_defaults["dataset_config"])
    parser.add_argument("--dataset-split", default=global_defaults["dataset_split"])
    parser.add_argument(
        "--hf-train-size",
        type=int,
        default=run_defaults["hf_train_size"],
        help="Number of HarmBench (HF) examples to train on after shuffle.",
    )
    parser.add_argument(
        "--csv-val-size",
        type=int,
        default=run_defaults["csv_val_size"],
        help="CSV behaviors used for periodic val ASR during training.",
    )
    parser.add_argument(
        "--csv-test-size",
        type=int,
        default=run_defaults["csv_test_size"],
        help="CSV behaviors used for baseline + final test eval.",
    )
    parser.add_argument(
        "--harmbench-csv-path",
        type=str,
        default=run_defaults.get("harmbench_csv_path", ""),
        help="HarmBench behaviors CSV (relative paths resolve from repo root).",
    )
    parser.add_argument(
        "--csv-seed",
        type=int,
        default=run_defaults.get("csv_seed", global_defaults["seed"]),
        help="Shuffle seed for the CSV val/test split.",
    )
    parser.add_argument("--seed", type=int, default=global_defaults["seed"])
    parser.add_argument("--results-dir", default=run_defaults["results_dir"])
    parser.add_argument(
        "--target-max-workers",
        type=int,
        default=run_defaults.get("target_max_workers", 16),
        help=(
            "Thread-pool size for concurrent target HTTP calls. Used during eval and during "
            "training rollouts (REINFORCE/RLOO batch): adversary stays sequential on this GPU "
            "while earlier target requests stay in flight on vLLM."
        ),
    )
    parser.add_argument(
        "--eval-method",
        choices=["judge", "heuristic"],
        default=run_defaults.get("eval_method", "judge"),
    )
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=run_defaults.get("refusal_threshold", 0.7),
    )
    parser.add_argument("--asr-threshold", type=float, default=run_defaults.get("asr_threshold", 0.3))
    parser.add_argument(
        "--adversary-prompt",
        choices=list(ADVERSARY_PROMPT_VARIANTS.keys()),
        default=run_defaults.get("adversary_prompt", "default"),
        help=(
            "Named rewriter preset for ASR comparisons. When not 'default', uses that variant "
            "unless --attacker-instruction is set. 'default' uses merged YAML seed when present."
        ),
    )
    parser.add_argument(
        "--attacker-instruction",
        default=None,
        help="Explicit rewriter instruction (overrides --adversary-prompt).",
    )
    parser.add_argument("--target-system-prompt", default=run_defaults["target_system_prompt"])
    parser.add_argument(
        "--adversary-policy",
        choices=["reinforce", "rloo"],
        default=run_defaults.get("adversary_policy", "reinforce"),
        help="Policy gradient: REINFORCE or RLOO (ignored when rejection sampling is enabled).",
    )
    parser.add_argument(
        "--adversary-reinforce-batch-size",
        type=int,
        default=run_defaults.get("adversary_reinforce_batch_size", 4),
        help="K adversary→target rollouts per step when not using rejection sampling (padded batch).",
    )
    parser.add_argument(
        "--rs-budget",
        type=int,
        default=run_defaults.get("rs_budget", 5),
        help="Max adversary samples per step when rejection sampling is enabled.",
    )
    parser.add_argument(
        "--rs-min-successes",
        type=int,
        default=run_defaults.get("rs_min_successes", 0),
        help=(
            "Rejection sampling: stop after this many successes (reward>0.5), up to --rs-budget. "
            "0 disables rejection sampling."
        ),
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        default=False,
        help=(
            "Train mode only: skip LoRA policy-gradient updates. "
            "Runs test-set eval once (baseline = final at init weights) for prompt comparison; omit for normal fine-tuning."
        ),
    )
    args = parser.parse_args()
    if args.rs_min_successes > 0 and args.adversary_policy == "rloo":
        parser.error(
            "Rejection sampling (--rs-min-successes > 0) requires --adversary-policy reinforce, not rloo."
        )
    return args


def _build_context(args: argparse.Namespace, defaults: dict[str, Any], device: str) -> RunContext:
    model_cfg = ModelConfig(model_id=args.adversary_model_id)
    adversary_cfg = UnslothAdversaryConfig(
        model_id=model_cfg.model_id,
        max_seq_length=model_cfg.max_seq_length,
        load_in_4bit=model_cfg.load_in_4bit,
        lora_r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
    )
    return RunContext(
        adversary_session=RuntimeCatalog.build_adversary_session(adversary_cfg),
        target_session=build_vllm_target_session(defaults),
        judge_session=RuntimeCatalog.build_judge_session(HarmbenchJudgeConfig()),
        device=device,
    )


def main() -> None:
    """Execute adversary-only training/eval and persist run artifacts."""
    run_start = time.time()
    defaults = load_default_config()
    args = parse_args(defaults)
    patch_run_args_from_config(defaults, args, run="adversary")
    sns.set_theme(style="whitegrid")
    run_defaults = defaults["runs"]["adversary"]

    args.attacker_instruction = resolve_adversary_attacker_instruction(
        cli_explicit=args.attacker_instruction,
        adversary_prompt=args.adversary_prompt,
        merged_yaml_instruction=run_defaults.get("attacker_instruction"),
    )
    if args.mode == "train":
        args.finetune = bool(run_defaults.get("finetune", True)) and not args.no_finetune
    else:
        args.finetune = True

    use_rs = args.rs_min_successes > 0
    policy_label = "RLOO" if args.adversary_policy == "rloo" else "REINFORCE"
    rs_note = " + rejection sampling" if use_rs else ""
    ft_note = "" if getattr(args, "finetune", True) else " | finetune=off"
    print(
        f"Adversary config: rewriter={args.adversary_prompt} | "
        f"optimizer={policy_label}{rs_note}{ft_note} | "
        f"rs_min_successes={args.rs_min_successes} rs_budget={args.rs_budget}"
    )

    train_cfg = AdversaryTrainingConfig(
        iterations=run_defaults["iterations"],
        lr=run_defaults["lr"],
        weight_decay=run_defaults["weight_decay"],
        eval_every=run_defaults["eval_every"],
        max_new_tokens=run_defaults["max_new_tokens"],
        target_max_new_tokens=int(run_defaults.get("target_max_new_tokens", 150)),
        training_csv_name=run_defaults.get("training_csv_name", "adversary_training_log.csv"),
        eval_csv_name=run_defaults.get("eval_csv_name", "adversary_eval_outputs.csv"),
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = EvaluationConfig(
        method=args.eval_method,
        refusal_threshold=args.refusal_threshold,
        asr_threshold=args.asr_threshold,
    )

    ctx = _build_context(args, defaults, device)

    hf_token = resolve_hf_token()
    train_data, _, _ = load_harmbench_subset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        train_size=args.hf_train_size,
        val_size=0,
        seed=args.seed,
        hf_token=hf_token,
    )
    prompts = build_prompt_pool(train_data, [])
    train_prompts = list(prompts)

    csv_raw = (args.harmbench_csv_path or "").strip()
    if not csv_raw:
        raise RuntimeError(
            "Set `runs.adversary.harmbench_csv_path` in config or pass --harmbench-csv-path "
            "(e.g. data/harmbench_behaviors.csv)."
        )
    resolved_csv = resolve_harmbench_csv_path(csv_raw, repo_root=_REPO_ROOT)
    if not resolved_csv.is_file():
        raise FileNotFoundError(f"HarmBench behaviors CSV not found: {resolved_csv}")
    args.harmbench_csv_resolved_path = str(resolved_csv)
    val_eval_examples, test_eval_examples = load_harmbench_csv_val_test_splits(
        resolved_csv,
        val_size=int(args.csv_val_size),
        test_size=int(args.csv_test_size),
        seed=int(args.csv_seed),
    )

    train_and_ft = args.mode == "train" and getattr(args, "finetune", True)
    test_note = (
        "CSV test (baseline + final) "
        if train_and_ft
        else "CSV test "
    )
    print(
        f"Dataset: HF train n={len(train_prompts)} | "
        f"CSV val (periodic) n={len(val_eval_examples)} | "
        f"{test_note}n={len(test_eval_examples)}"
    )

    baseline_eval, baseline_rows, baseline_metrics = _eval_suite(
        test_eval_examples,
        args.attacker_instruction,
        args.target_system_prompt,
        train_cfg,
        eval_cfg,
        ctx,
        args,
    )
    if train_and_ft:
        print("Baseline metrics:", baseline_metrics)
    else:
        print("Test eval metrics:", baseline_metrics)

    training_rows: list[dict[str, Any]] = []
    final_eval = baseline_eval
    final_eval_rows = baseline_rows
    final_metrics = baseline_metrics
    model = ctx.adversary_session.runtime
    optimizer: torch.optim.Optimizer | None = None
    if args.mode == "train" and args.finetune:
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    eval_every = max(1, int(train_cfg.eval_every))

    if args.mode == "train" and args.finetune:
        rs_budget = max(1, int(args.rs_budget))
        for iteration in range(train_cfg.iterations):
            idx = torch.randint(0, len(train_prompts), ()).item()
            prompt = train_prompts[idx]

            batch_gen_ids: list[torch.Tensor] = []
            batch_rewards: list[float] = []
            batch_prompt_lens: list[int] = []
            adv_texts: list[str] = []
            target_resps: list[str] = []
            verdicts: list[str] = []

            if use_rs:
                n_successes_so_far = 0
                for _ in range(rs_budget):
                    sampled, _ = adversary_output(
                        prompt=prompt,
                        attacker_instruction=args.attacker_instruction,
                        adversary_session=ctx.adversary_session,
                        device=ctx.device,
                        max_new_tokens=train_cfg.max_new_tokens,
                    )
                    adversary_text = sampled["completion_text"].strip()
                    target_response, _ = target_generate(
                        prompt=adversary_text,
                        target_session=ctx.target_session,
                        device=ctx.device,
                        target_system_prompt=args.target_system_prompt,
                        max_new_tokens=train_cfg.target_max_new_tokens,
                    )
                    reward, verdict = reward_from_response(prompt, target_response, eval_cfg, ctx)
                    batch_gen_ids.append(sampled["gen_ids"])
                    batch_rewards.append(reward)
                    batch_prompt_lens.append(sampled["prompt_len"])
                    adv_texts.append(adversary_text)
                    target_resps.append(target_response)
                    verdicts.append(verdict)
                    if reward > 0.5:
                        n_successes_so_far += 1
                        if n_successes_so_far >= args.rs_min_successes:
                            break
            else:
                batch_size = max(1, int(args.adversary_reinforce_batch_size))
                train_target_pool = getattr(
                    ctx.target_session.runtime, "supports_concurrent_target_inference", False
                )
                train_workers = cap_thread_workers(batch_size, args.target_max_workers)
                if not train_target_pool:
                    for _ in range(batch_size):
                        sampled, _ = adversary_output(
                            prompt=prompt,
                            attacker_instruction=args.attacker_instruction,
                            adversary_session=ctx.adversary_session,
                            device=ctx.device,
                            max_new_tokens=train_cfg.max_new_tokens,
                        )
                        adversary_text = sampled["completion_text"].strip()
                        target_response, _ = target_generate(
                            prompt=adversary_text,
                            target_session=ctx.target_session,
                            device=ctx.device,
                            target_system_prompt=args.target_system_prompt,
                            max_new_tokens=train_cfg.target_max_new_tokens,
                        )
                        reward, verdict = reward_from_response(prompt, target_response, eval_cfg, ctx)
                        batch_gen_ids.append(sampled["gen_ids"])
                        batch_rewards.append(reward)
                        batch_prompt_lens.append(sampled["prompt_len"])
                        adv_texts.append(adversary_text)
                        target_resps.append(target_response)
                        verdicts.append(verdict)
                else:
                    pending_rollouts: list[
                        tuple[dict[str, Any], str, Future[tuple[str, float]]]
                    ] = []
                    with ThreadPoolExecutor(max_workers=train_workers) as executor:
                        for _ in range(batch_size):
                            sampled, _ = adversary_output(
                                prompt=prompt,
                                attacker_instruction=args.attacker_instruction,
                                adversary_session=ctx.adversary_session,
                                device=ctx.device,
                                max_new_tokens=train_cfg.max_new_tokens,
                            )
                            adversary_text = sampled["completion_text"].strip()
                            request = GenerationRequest(
                                system_prompt=args.target_system_prompt,
                                user_prompt=adversary_text,
                                max_new_tokens=train_cfg.target_max_new_tokens,
                                temperature=0.0,
                            )
                            fut = executor.submit(
                                timed_target_generate, ctx.target_session, ctx.device, request
                            )
                            pending_rollouts.append((sampled, adversary_text, fut))
                        for sampled, adversary_text, fut in pending_rollouts:
                            target_response, _ = fut.result()
                            reward, verdict = reward_from_response(prompt, target_response, eval_cfg, ctx)
                            batch_gen_ids.append(sampled["gen_ids"])
                            batch_rewards.append(reward)
                            batch_prompt_lens.append(sampled["prompt_len"])
                            adv_texts.append(adversary_text)
                            target_resps.append(target_response)
                            verdicts.append(verdict)

            tok = model.tokenizer
            pad_id = tok.pad_token_id
            if pad_id is None:
                pad_id = tok.eos_token_id
            if pad_id is None:
                raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id for batch padding.")
            padded_ids, valid_lens = pad_gen_ids_batch(batch_gen_ids, int(pad_id))
            device_t = padded_ids.device
            rewards_t = torch.tensor(batch_rewards, dtype=torch.float32, device=device_t)

            if use_rs:
                loss_val, _n_used = rejection_sampling_update_sgd(
                    model=model,
                    optimizer=optimizer,  # type: ignore[arg-type]
                    gen_ids=padded_ids,
                    prompt_lens=batch_prompt_lens,
                    rewards=rewards_t,
                    valid_seq_lens=valid_lens,
                )
                if loss_val is None:
                    loss_val = float("nan")
            elif args.adversary_policy == "rloo":
                loss_val, _ = rloo_update_batch_sgd(
                    model=model,
                    optimizer=optimizer,  # type: ignore[arg-type]
                    gen_ids=padded_ids,
                    prompt_lens=batch_prompt_lens,
                    rewards=rewards_t,
                    valid_seq_lens=valid_lens,
                )
            else:
                loss_val, _ = reinforce_update_batch_sgd(
                    model=model,
                    optimizer=optimizer,  # type: ignore[arg-type]
                    gen_ids=padded_ids,
                    prompt_lens=batch_prompt_lens,
                    rewards=rewards_t,
                    valid_seq_lens=valid_lens,
                )

            mean_r = float(rewards_t.mean().item())
            log_batch = len(batch_rewards)

            eval_asr: float | str = ""
            eval_refusal_rate: float | str = ""
            if iteration % eval_every == 0:
                periodic_eval, _, periodic_metrics = _eval_suite(
                    val_eval_examples,
                    args.attacker_instruction,
                    args.target_system_prompt,
                    train_cfg,
                    eval_cfg,
                    ctx,
                    args,
                )
                eval_asr = periodic_eval.asr
                eval_refusal_rate = periodic_eval.refusal_rate
                print(
                    f"Iter {iteration} | val ASR: {periodic_eval.asr:.3f} | "
                    f"refusal_rate: {periodic_eval.refusal_rate:.3f} | "
                    f"latency_ms_mean: {periodic_metrics['latency_ms_mean']:.1f}"
                )

            training_rows.append(
                {
                    "iteration": iteration,
                    "dataset_index": idx,
                    "prompt": clean_text(prompt),
                    "adversary_output": clean_text(adv_texts[0]),
                    "target_response": clean_text(target_resps[0]),
                    "reward": mean_r,
                    "loss": loss_val,
                    "verdict": verdicts[0],
                    "max_reward": max(batch_rewards),
                    "batch_size": log_batch,
                    "adversary_policy": args.adversary_policy,
                    "adversary_prompt": args.adversary_prompt,
                    "rejection_sampling": use_rs,
                    "eval_asr": eval_asr,
                    "eval_refusal_rate": eval_refusal_rate,
                }
            )

        final_eval, final_eval_rows, final_metrics = _eval_suite(
            test_eval_examples,
            args.attacker_instruction,
            args.target_system_prompt,
            train_cfg,
            eval_cfg,
            ctx,
            args,
        )
        print("Final test metrics:", final_metrics)

    elif args.mode == "train" and not args.finetune:
        print(
            "Skipping fine-tuning (--no-finetune): baseline/final = test eval at init weights; "
            "no periodic val eval (no training iterations)."
        )

    results_dir = Path(args.results_dir).resolve()
    # After successful fine-tuning, LoRA weights are written to results_dir/adapters (no separate CLI).
    auto_adapters = args.mode == "train" and getattr(args, "finetune", True)
    adapter_path = _try_save_adapters_print(ctx, "adapters" if auto_adapters else None, results_dir)
    run_seconds = time.time() - run_start
    baseline_df = pd.DataFrame(baseline_rows)
    final_df = pd.DataFrame(final_eval_rows)
    training_df = pd.DataFrame(training_rows)

    save_artifacts(
        args=args,
        results_dir=results_dir,
        train_cfg=train_cfg,
        baseline_metrics=baseline_metrics,
        final_metrics=final_metrics,
        baseline_df=baseline_df,
        final_df=final_df,
        training_df=training_df,
        run_seconds=run_seconds,
        adapter_path=adapter_path,
        config_snapshot=build_config_snapshot(defaults, cli_args=args),
    )


if __name__ == "__main__":
    main()
