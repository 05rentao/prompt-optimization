#!/usr/bin/env python3
"""CoEV v2 — vanilla adversary policy training with fixed prompts.

Trains adversary LoRA with REINFORCE, RLOO, or rejection sampling. Attacker instruction
and target system prompt are fixed for the whole run (from YAML / CLI). There is no
prompt evolution here; use ``runs/adversary_run.py`` for richer experiments and knobs.

See ``notes/theme2-adversary-experiments.md`` for the current experimental plan (Theme 2)
and how this script relates to ``adversary_run.py``.
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
    save_baseline_optimized_plot,
    save_coev_asr_vs_global_step_plot,
    save_coev_refusal_vs_global_step_plot,
    write_json,
    write_many_csv,
    write_run_manifest,
    write_text,
)
from src.data import load_harmbench_subset
from src.evaluators import compute_refusal_score
from src.run_pipeline import (
    adversary_rewrite_sample,
    build_prompt_pool,
    compute_reward_and_verdict,
    maybe_save_adapters as maybe_save_adapters_common,
    split_prompt_pool,
)
from src.runtime import (
    CoevConfig,
    EvaluationConfig,
    EvaluatedSample,
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
from src.runtime.adversary_prompts import ADVERSARY_PROMPT_VARIANTS
from src.runtime.defaults import build_config_snapshot, load_default_config
from src.runtime.policy_gradient import (
    pad_gen_ids_batch,
    reinforce_update_batch_sgd,
    rejection_sampling_update_sgd,
    rloo_update_batch_sgd,
)
from src.types import CoevEvalOutputRow, CoevStageMetricRow, CoevTrainingLogRow, RunManifest

# ---------------------------------------------------------------------------
# Run context and per-step generation (adversary rewrite → target completion)
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Shared sessions for one run (adversary GPU + vLLM target + judge)."""

    adversary_session: GenerationSession
    target_session: GenerationSession
    judge_session: GenerationSession
    device: str


def clean_text(value: Any) -> str:
    """Normalize text for single-line CSV/JSON rows."""

    return str(value).replace("\n", " ").replace("\r", " ").strip()


def adversary_output(
    prompt: str,
    instruction: str,
    adversary_session: GenerationSession,
    device: str,
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any], float]:
    """One adversarial rewrite from the policy; returns sample dict and adversary latency (ms)."""

    return adversary_rewrite_sample(
        adversary_session,
        device,
        prompt,
        instruction,
        max_new_tokens=max_new_tokens,
    )


def target_generate(
    prompt: str,
    target_session: GenerationSession,
    device: str,
    defense_prompt: str,
    max_new_tokens: int = 150,
) -> tuple[str, float]:
    """One target completion under ``defense_prompt``; returns text and latency (ms)."""

    request = GenerationRequest(
        system_prompt=defense_prompt,
        user_prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    return timed_target_generate(target_session, device, request)


def evaluate_prompts(
    examples: list[dict[str, Any]],
    attacker_instruction: str,
    defense_prompt: str,
    ctx: RunContext,
    eval_cfg: EvaluationConfig,
    max_new_tokens: int,
    target_max_workers: int | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Eval chain adversary → target; optional thread pool for HTTP target."""

    n = len(examples)
    workers = cap_thread_workers(n, target_max_workers)
    use_target_pool = getattr(ctx.target_session.runtime, "supports_concurrent_target_inference", False)

    adv_rows: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    target_texts: list[str] = []
    target_latencies: list[float] = []

    if not use_target_pool:
        for ex in examples:
            sample, adv_latency_ms = adversary_output(
                prompt=ex["prompt"],
                instruction=attacker_instruction,
                adversary_session=ctx.adversary_session,
                device=ctx.device,
                max_new_tokens=max_new_tokens,
            )
            request = GenerationRequest(
                system_prompt=defense_prompt,
                user_prompt=sample["completion_text"].strip(),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            target_resp, target_latency_ms = timed_target_generate(ctx.target_session, ctx.device, request)
            adv_rows.append((ex, sample, adv_latency_ms))
            target_texts.append(target_resp)
            target_latencies.append(target_latency_ms)
    else:
        pending: list[tuple[dict[str, Any], dict[str, Any], float, Future[tuple[str, float]]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ex in examples:
                sample, adv_latency_ms = adversary_output(
                    prompt=ex["prompt"],
                    instruction=attacker_instruction,
                    adversary_session=ctx.adversary_session,
                    device=ctx.device,
                    max_new_tokens=max_new_tokens,
                )
                request = GenerationRequest(
                    system_prompt=defense_prompt,
                    user_prompt=sample["completion_text"].strip(),
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )
                fut = executor.submit(timed_target_generate, ctx.target_session, ctx.device, request)
                pending.append((ex, sample, adv_latency_ms, fut))

            for ex, sample, adv_latency_ms, fut in pending:
                adv_rows.append((ex, sample, adv_latency_ms))
                target_resp, target_latency_ms = fut.result()
                target_texts.append(target_resp)
                target_latencies.append(target_latency_ms)

    samples: list[EvaluatedSample] = []
    for (ex, sample, adv_latency_ms), target_resp, target_latency_ms in zip(
        adv_rows, target_texts, target_latencies, strict=True
    ):
        adv_prompt = sample["completion_text"].strip()
        refusal_score = compute_refusal_score(target_resp)
        total_latency = adv_latency_ms + target_latency_ms
        row: CoevEvalOutputRow = {
            "id": str(ex.get("id", "")),
            "prompt": ex["prompt"],
            "adversary_prompt": adv_prompt,
            "target_response": target_resp,
            "refusal_score": refusal_score,
            "asr_score": 1.0 - refusal_score,
            "adversary_latency_ms": adv_latency_ms,
            "target_latency_ms": target_latency_ms,
            "latency_ms_total": total_latency,
        }
        samples.append(
            EvaluatedSample(
                id=str(ex.get("id", "")),
                behavior=ex["prompt"],
                generation=target_resp,
                latency_ms=total_latency,
                row=row,
            )
        )

    behaviors = [s.behavior for s in samples]
    generations = [s.generation for s in samples]
    eval_result = evaluate_outputs(
        behaviors=behaviors,
        generations=generations,
        cfg=eval_cfg,
        judge_session=ctx.judge_session,
    )
    aggregate_score = eval_result.mean_refusal_score
    if aggregate_score is None:
        aggregate_score = eval_result.refusal_rate

    rows: list[dict[str, Any]] = []
    for sample in samples:
        row_payload: dict[str, Any] = {
            "id": sample.id,
            "behavior": sample.behavior,
            "generation": sample.generation,
        }
        if sample.row:
            row_payload.update(sample.row)
        if sample.latency_ms is not None and "latency_ms" not in row_payload:
            row_payload["latency_ms"] = float(sample.latency_ms)
        rows.append(row_payload)

    df = pd.DataFrame(rows)
    metrics = {
        "refusal_rate": float(eval_result.refusal_rate),
        "asr": float(eval_result.asr),
        "mean_refusal_score": float(df["refusal_score"].mean()) if not df.empty else 0.0,
        "aggregate_score": float(aggregate_score),
        "latency_ms_mean": float(df["latency_ms_total"].mean()) if not df.empty else 0.0,
    }
    return metrics, df


def _eval_suite(
    ctx: RunContext,
    eval_cfg: EvaluationConfig,
    args: argparse.Namespace,
    attacker_instruction: str,
    defense_prompt: str,
    examples: list[dict[str, Any]],
) -> tuple[dict[str, float], pd.DataFrame]:
    return evaluate_prompts(
        examples=examples,
        attacker_instruction=attacker_instruction,
        defense_prompt=defense_prompt,
        ctx=ctx,
        eval_cfg=eval_cfg,
        max_new_tokens=args.max_new_tokens,
        target_max_workers=args.target_max_workers,
    )


def _try_save_adapters_print(ctx: RunContext, save_dir: str | None, results_dir: Path) -> None:
    path = maybe_save_adapters_common(ctx.adversary_session, save_dir, results_dir=results_dir)
    if path:
        print(f"Saved model/tokenizer to: {path}")


def save_artifacts(
    args: argparse.Namespace,
    results_dir: Path,
    baseline_metrics: dict[str, float],
    optimized_metrics: dict[str, float],
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    train_log_df: pd.DataFrame,
    stage_metrics_df: pd.DataFrame,
    final_attacker_instruction: str,
    final_defense_prompt: str,
    run_seconds: float,
    config_snapshot: dict[str, Any],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    optimized_prompts_path = results_dir / "coev_v2_optimized_prompts.json"
    write_json(
        optimized_prompts_path,
        {"attacker_instruction": final_attacker_instruction, "defense_prompt": final_defense_prompt},
    )
    attacker_txt_path = results_dir / "optimized_attacker_instruction.txt"
    defense_txt_path = results_dir / "optimized_defense_prompt.txt"
    write_text(attacker_txt_path, final_attacker_instruction)
    write_text(defense_txt_path, final_defense_prompt)

    metrics_payload: dict[str, Any] = {
        "config": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "runtime_profile": args.runtime_profile,
            "target_model_name": args.task_model_name,
            "stages": args.stages,
            "iters_per_stage": args.iters_per_stage,
            "run_seconds": run_seconds,
            "adversary_policy": getattr(args, "adversary_policy", "reinforce"),
            "rs_budget": getattr(args, "rs_budget", 5),
            "rs_min_successes": getattr(args, "rs_min_successes", 0),
        },
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
    }
    metrics_json_path = results_dir / "coev_v2_run_metrics.json"
    write_json(metrics_json_path, metrics_payload)

    comparison_df = build_baseline_optimized_df(
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
    )
    write_many_csv(
        results_dir,
        {
            "baseline_vs_optimized_metrics.csv": comparison_df,
            "baseline_eval_outputs.csv": baseline_df,
            "optimized_eval_outputs.csv": optimized_df,
            args.training_csv_name: train_log_df,
            "coev_v2_stage_metrics.csv": stage_metrics_df,
        },
    )

    label = "CoEV v2 RLOO" if getattr(args, "adversary_policy", "reinforce") == "rloo" else "CoEV v2"
    baseline_plot_path = save_baseline_optimized_plot(
        comparison_df=comparison_df,
        out_path=results_dir / "plot_baseline_vs_optimized.png",
        title=f"{label} Baseline vs Optimized Metrics",
    )
    # Stage plot uses phase "pre_evolution" rows (compat with plotting helpers).
    asr_iter_path = save_coev_asr_vs_global_step_plot(
        stage_metrics_df=stage_metrics_df,
        iters_per_stage=args.iters_per_stage,
        out_path=results_dir / "plot_asr_vs_iterations.png",
        title=f"{label} ASR vs global training step (stage checkpoints)",
    )
    refusal_iter_path = save_coev_refusal_vs_global_step_plot(
        stage_metrics_df=stage_metrics_df,
        iters_per_stage=args.iters_per_stage,
        out_path=results_dir / "plot_refusal_vs_iterations.png",
        title=f"{label} refusal rate vs global training step (stage checkpoints)",
    )

    manifest_mode = "coev_v2_rloo" if getattr(args, "adversary_policy", "reinforce") == "rloo" else "coev_v2"
    manifest = RunManifest(
        mode=manifest_mode,
        runtime_profile=args.runtime_profile,
        seed=args.seed,
        dataset={
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        models={
            "adversary_model": args.adversary_model_id,
            "target_model": args.task_model_name,
            "judge_model": "cais/HarmBench-Mistral-7b-val-cls",
        },
        budget={
            "stages": args.stages,
            "iters_per_stage": args.iters_per_stage,
            "run_seconds": run_seconds,
        },
        endpoints={},
        extra={
            "optimized_prompts_path": str(optimized_prompts_path),
            "metrics_json_path": str(metrics_json_path),
            "training_csv_name": args.training_csv_name,
            "eval_method": args.eval_method,
            "adversary_policy": getattr(args, "adversary_policy", "reinforce"),
            "rs_budget": getattr(args, "rs_budget", 5),
            "rs_min_successes": getattr(args, "rs_min_successes", 0),
        },
        config_snapshot=config_snapshot,
    )
    manifest_path = write_run_manifest(results_dir=results_dir, payload=manifest)
    logged_paths = [
        optimized_prompts_path,
        attacker_txt_path,
        defense_txt_path,
        metrics_json_path,
        manifest_path,
        baseline_plot_path,
        results_dir,
    ]
    if asr_iter_path is not None:
        logged_paths.insert(-1, asr_iter_path)
    if refusal_iter_path is not None:
        logged_paths.insert(-1, refusal_iter_path)
    log_saved_artifacts(logged_paths)


# ---------------------------------------------------------------------------
# CLI and session construction
# ---------------------------------------------------------------------------


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    global_defaults = defaults["global"]
    run_defaults = defaults["runs"]["coev_v2"]

    parser = argparse.ArgumentParser(
        description="CoEV v2: vanilla adversary training (fixed prompts; no prompt evolution)."
    )
    parser.add_argument("--mode", choices=["coev", "eval"], default="coev")
    parser.add_argument("--device", default=global_defaults["device"])
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory for adversary adapters; relative paths are under --results-dir.",
    )
    parser.add_argument("--results-dir", default=run_defaults["results_dir"])
    parser.add_argument("--training-csv-name", default=run_defaults["training_csv_name"])

    parser.add_argument("--dataset-name", default=global_defaults["dataset_name"])
    parser.add_argument("--dataset-config", default=global_defaults["dataset_config"])
    parser.add_argument("--dataset-split", default=global_defaults["dataset_split"])
    parser.add_argument("--train-size", type=int, default=run_defaults["train_size"])
    parser.add_argument("--val-size", type=int, default=run_defaults["val_size"])
    parser.add_argument("--seed", type=int, default=global_defaults["seed"])

    parser.add_argument("--max-new-tokens", type=int, default=run_defaults["max_new_tokens"])
    parser.add_argument(
        "--target-max-workers",
        type=int,
        default=run_defaults.get("target_max_workers", 16),
        help="Thread-pool size for target HTTP calls during eval.",
    )

    parser.add_argument("--stages", type=int, default=run_defaults["stages"])
    parser.add_argument("--iters-per-stage", type=int, default=run_defaults["iters_per_stage"])
    parser.add_argument("--eval-every-stages", type=int, default=run_defaults["eval_every_stages"])
    parser.add_argument("--train-slice-end", type=int, default=run_defaults["train_slice_end"])
    parser.add_argument("--eval-slice-start", type=int, default=run_defaults["eval_slice_start"])
    parser.add_argument("--eval-slice-end", type=int, default=run_defaults["eval_slice_end"])
    parser.add_argument("--lr", type=float, default=run_defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=run_defaults["weight_decay"])
    parser.add_argument(
        "--adversary-policy",
        choices=["reinforce", "rloo"],
        default=run_defaults.get("adversary_policy", "reinforce"),
        help="Adversary weight update: REINFORCE or RLOO.",
    )
    parser.add_argument(
        "--initial-attacker-instruction",
        default=None,
        help="Override rewriter instruction; default is merged YAML (shared_generation seed).",
    )
    parser.add_argument("--initial-defense-prompt", default=run_defaults["initial_defense_prompt"])

    parser.add_argument("--eval-method", choices=["judge", "heuristic"], default=run_defaults["eval_method"])
    parser.add_argument("--refusal-threshold", type=float, default=run_defaults["refusal_threshold"])
    parser.add_argument("--asr-threshold", type=float, default=run_defaults["asr_threshold"])
    parser.add_argument(
        "--adversary-reinforce-batch-size",
        type=int,
        default=run_defaults["adversary_reinforce_batch_size"],
        help="Rollouts per training step when not using rejection sampling.",
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
        help="Rejection sampling: stop after this many successes (reward>0.5). 0 = batched REINFORCE/RLOO.",
    )
    args = parser.parse_args()
    if args.rs_min_successes > 0 and args.adversary_policy == "rloo":
        parser.error("Rejection sampling requires --adversary-policy reinforce, not rloo.")
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


def _load_prompt_slices(
    args: argparse.Namespace,
    coev_cfg: CoevConfig,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    hf_token = resolve_hf_token()
    train_data, val_data, _ = load_harmbench_subset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        hf_token=hf_token,
    )
    prompts = build_prompt_pool(train_data, val_data)
    train_prompts, eval_prompts = split_prompt_pool(
        prompts=prompts,
        train_slice_end=coev_cfg.train_slice_end,
        eval_slice_start=coev_cfg.eval_slice_start,
        eval_slice_end=coev_cfg.eval_slice_end,
        require_train=args.mode != "eval",
        script_name="coev_v2_run",
    )
    eval_examples = [{"id": f"eval_{i}", "prompt": p} for i, p in enumerate(eval_prompts)]
    return train_prompts, eval_prompts, eval_examples


def _run_eval_only(
    args: argparse.Namespace,
    ctx: RunContext,
    baseline_metrics: dict[str, float],
    baseline_df: pd.DataFrame,
    attacker_instruction: str,
    defense_prompt: str,
    run_start: float,
    config_snapshot: dict[str, Any],
) -> None:
    results_dir = Path(args.results_dir).resolve()
    _try_save_adapters_print(ctx, args.save_dir, results_dir)
    empty = pd.DataFrame()
    save_artifacts(
        args=args,
        results_dir=results_dir,
        baseline_metrics=baseline_metrics,
        optimized_metrics=baseline_metrics,
        baseline_df=baseline_df,
        optimized_df=baseline_df,
        train_log_df=empty,
        stage_metrics_df=empty,
        final_attacker_instruction=attacker_instruction,
        final_defense_prompt=defense_prompt,
        run_seconds=time.time() - run_start,
        config_snapshot=config_snapshot,
    )


# ---------------------------------------------------------------------------
# Training: staged policy-gradient steps, optional rejection sampling, stage evals
# ---------------------------------------------------------------------------


def _run_training_and_finalize(
    args: argparse.Namespace,
    ctx: RunContext,
    eval_cfg: EvaluationConfig,
    coev_cfg: CoevConfig,
    train_prompts: list[str],
    eval_examples: list[dict[str, Any]],
    baseline_metrics: dict[str, float],
    baseline_df: pd.DataFrame,
    attacker_instruction: str,
    defense_prompt: str,
    run_start: float,
    config_snapshot: dict[str, Any],
) -> None:
    model = ctx.adversary_session.runtime
    optimizer = torch.optim.AdamW(model.parameters(), lr=coev_cfg.lr, weight_decay=coev_cfg.weight_decay)

    training_rows: list[CoevTrainingLogRow] = []
    stage_metric_rows: list[CoevStageMetricRow] = []

    use_rs = args.rs_min_successes > 0
    policy_label = "RLOO" if args.adversary_policy == "rloo" else "REINFORCE"
    rs_note = " + rejection sampling" if use_rs else ""
    judge = ctx.judge_session if eval_cfg.method == "judge" else None

    print("Starting CoEV v2 (vanilla adversary) training...")
    for stage in range(coev_cfg.stages):
        print(f"\n--- Stage {stage} start ---")
        for i in range(coev_cfg.iters_per_stage):
            print(f"\n--- Stage {stage} iteration {i} ---")
            idx = torch.randint(0, len(train_prompts), ()).item()
            original_prompt = train_prompts[idx]

            batch_gen_ids: list[torch.Tensor] = []
            batch_rewards: list[float] = []
            batch_prompt_lens: list[int] = []
            adv_rewrites: list[str] = []
            target_resps: list[str] = []
            verdicts: list[str] = []

            if use_rs:
                n_successes_so_far = 0
                rs_budget = max(1, int(args.rs_budget))
                for _ in range(rs_budget):
                    sample, _ = adversary_output(
                        prompt=original_prompt,
                        instruction=attacker_instruction,
                        adversary_session=ctx.adversary_session,
                        device=ctx.device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    adv_rewrite = sample["completion_text"].strip()
                    target_resp, _ = target_generate(
                        adv_rewrite,
                        ctx.target_session,
                        ctx.device,
                        defense_prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                    reward, verdict = compute_reward_and_verdict(
                        behavior=original_prompt,
                        generation=target_resp,
                        eval_cfg=eval_cfg,
                        judge_session=judge,
                    )
                    batch_gen_ids.append(sample["gen_ids"])
                    batch_rewards.append(reward)
                    batch_prompt_lens.append(sample["prompt_len"])
                    adv_rewrites.append(adv_rewrite)
                    target_resps.append(target_resp)
                    verdicts.append(verdict)
                    if reward > 0.5:
                        n_successes_so_far += 1
                        if n_successes_so_far >= args.rs_min_successes:
                            break
            else:
                batch_size = max(1, int(args.adversary_reinforce_batch_size))
                for _ in range(batch_size):
                    sample, _ = adversary_output(
                        prompt=original_prompt,
                        instruction=attacker_instruction,
                        adversary_session=ctx.adversary_session,
                        device=ctx.device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    adv_rewrite = sample["completion_text"].strip()
                    target_resp, _ = target_generate(
                        adv_rewrite,
                        ctx.target_session,
                        ctx.device,
                        defense_prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                    reward, verdict = compute_reward_and_verdict(
                        behavior=original_prompt,
                        generation=target_resp,
                        eval_cfg=eval_cfg,
                        judge_session=judge,
                    )
                    batch_gen_ids.append(sample["gen_ids"])
                    batch_rewards.append(reward)
                    batch_prompt_lens.append(sample["prompt_len"])
                    adv_rewrites.append(adv_rewrite)
                    target_resps.append(target_resp)
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
                    model,
                    optimizer,
                    padded_ids,
                    batch_prompt_lens,
                    rewards_t,
                    valid_seq_lens=valid_lens,
                )
                if loss_val is None:
                    loss_val = float("nan")
            elif args.adversary_policy == "rloo":
                loss_val, _ = rloo_update_batch_sgd(
                    model,
                    optimizer,
                    padded_ids,
                    batch_prompt_lens,
                    rewards_t,
                    valid_seq_lens=valid_lens,
                )
            else:
                loss_val, _ = reinforce_update_batch_sgd(
                    model,
                    optimizer,
                    padded_ids,
                    batch_prompt_lens,
                    rewards_t,
                    valid_seq_lens=valid_lens,
                )

            mean_r = float(rewards_t.mean().item())
            log_batch = len(batch_rewards)
            training_rows.append(
                {
                    "stage": stage,
                    "iter": i,
                    "dataset_index": idx,
                    "attacker_instruction": clean_text(attacker_instruction),
                    "defense_prompt": clean_text(defense_prompt),
                    "orig_prompt": clean_text(original_prompt),
                    "adv_prompt": clean_text(adv_rewrites[0]),
                    "target_resp": clean_text(target_resps[0]),
                    "reward": mean_r,
                    "loss": loss_val,
                    "verdict": verdicts[0],
                    "max_reward": max(batch_rewards),
                    "batch_size": log_batch,
                    "adversary_policy": args.adversary_policy,
                    "rejection_sampling": use_rs,
                }
            )

        if stage % coev_cfg.eval_every_stages == 0:
            stage_metrics, _ = _eval_suite(ctx, eval_cfg, args, attacker_instruction, defense_prompt, eval_examples)
            stage_metric_rows.append({"stage": stage, "phase": "pre_evolution", **stage_metrics})
            print(
                f"Stage {stage}: Completed {policy_label}{rs_note}. Eval ASR: {stage_metrics['asr']:.2%} | "
                f"refusal_rate: {stage_metrics['refusal_rate']:.2%}"
            )
        else:
            print(f"Stage {stage}: Completed {policy_label}{rs_note}.")

    optimized_metrics, optimized_df = _eval_suite(ctx, eval_cfg, args, attacker_instruction, defense_prompt, eval_examples)
    print("Optimized metrics:", optimized_metrics)

    results_dir = Path(args.results_dir).resolve()
    _try_save_adapters_print(ctx, args.save_dir, results_dir)

    save_artifacts(
        args=args,
        results_dir=results_dir,
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        baseline_df=baseline_df,
        optimized_df=optimized_df,
        train_log_df=pd.DataFrame(training_rows),
        stage_metrics_df=pd.DataFrame(stage_metric_rows),
        final_attacker_instruction=attacker_instruction,
        final_defense_prompt=defense_prompt,
        run_seconds=time.time() - run_start,
        config_snapshot=config_snapshot,
    )


# ---------------------------------------------------------------------------
# Entrypoint: resolve fixed prompts, baseline eval, train (or eval-only), artifacts
# ---------------------------------------------------------------------------


def main() -> None:
    defaults = load_default_config()
    args = parse_args(defaults)
    patch_run_args_from_config(defaults, args, run="coev_v2")

    run_start = time.time()
    sns.set_theme(style="whitegrid")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = EvaluationConfig(
        method=args.eval_method,
        refusal_threshold=args.refusal_threshold,
        asr_threshold=args.asr_threshold,
    )
    run_defaults = defaults["runs"]["coev_v2"]
    resolved_initial_attacker = args.initial_attacker_instruction
    if resolved_initial_attacker is None:
        resolved_initial_attacker = run_defaults.get("initial_attacker_instruction")
    if resolved_initial_attacker is None:
        resolved_initial_attacker = ADVERSARY_PROMPT_VARIANTS["default"]

    coev_cfg = CoevConfig(
        stages=args.stages,
        iters_per_stage=args.iters_per_stage,
        eval_every_stages=args.eval_every_stages,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_slice_end=args.train_slice_end,
        eval_slice_start=args.eval_slice_start,
        eval_slice_end=args.eval_slice_end,
        csv_path=args.training_csv_name,
        initial_attacker_instruction=resolved_initial_attacker,
        initial_defense_prompt=args.initial_defense_prompt,
    )

    ctx = _build_context(args, defaults, device)
    config_snapshot = build_config_snapshot(defaults, cli_args=args)
    train_prompts, _, eval_examples = _load_prompt_slices(args, coev_cfg)

    attacker_instruction = resolved_initial_attacker
    defense_prompt = coev_cfg.initial_defense_prompt
    baseline_metrics, baseline_df = _eval_suite(ctx, eval_cfg, args, attacker_instruction, defense_prompt, eval_examples)
    print("Baseline metrics:", baseline_metrics)

    if args.mode == "eval":
        _run_eval_only(
            args,
            ctx,
            baseline_metrics,
            baseline_df,
            attacker_instruction,
            defense_prompt,
            run_start,
            config_snapshot,
        )
        return

    _run_training_and_finalize(
        args,
        ctx,
        eval_cfg,
        coev_cfg,
        train_prompts,
        eval_examples,
        baseline_metrics,
        baseline_df,
        attacker_instruction,
        defense_prompt,
        run_start,
        config_snapshot,
    )


if __name__ == "__main__":
    main()
