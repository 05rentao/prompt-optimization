#!/usr/bin/env python3
"""CoEV v2 runner with staged REINFORCE + GEPA-based prompt evolution.

This runner preserves the stage-based coevolution workflow from `runs/coev_run.py`
while replacing handcrafted prompt-evolution logic with GEPA optimization loops
similar to `src/runtime/gepa_prompt_optimization.py`.
"""

from __future__ import annotations

import unsloth  # noqa: F401 — must import before transformers for Unsloth patches

import argparse
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as f

from src.artifacts import (
    build_baseline_optimized_df,
    log_saved_artifacts,
    save_baseline_optimized_plot,
    save_coev_asr_vs_global_step_plot,
    save_coev_refusal_vs_global_step_plot,
    save_trajectory_plot,
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
    build_reflection_gateway_for_defaults,
    build_vllm_target_session,
    cap_thread_workers,
    evaluate_outputs,
    resolve_hf_token,
    resolve_reflection_env_overrides,
    timed_target_generate,
)
from src.runtime.defaults import build_config_snapshot, load_default_config, merged_run_defaults
from src.runtime.openai_reflection_gateway import OpenAIReflectionGateway
from src.runtime.gepa_prompt_optimization import (
    DualRoleGepaContext,
    DualRoleGepaPromptOptimizationConfig,
    run_dual_role_gepa_prompt_optimization,
)
from src.types import CoevEvalOutputRow, CoevStageMetricRow, CoevTrainingLogRow, OptimizerTraceRow, RunManifest


@dataclass
class RunContext:
    """Shared long-lived sessions and runtime dependencies for one run."""

    adversary_session: GenerationSession
    target_session: GenerationSession
    judge_session: GenerationSession
    reflection_gateway: OpenAIReflectionGateway
    device: str


def clean_text(value: Any) -> str:
    """Normalize values for single-line CSV/JSON logging."""

    return str(value).replace("\n", " ").replace("\r", " ").strip()


def adversary_output(
    prompt: str,
    instruction: str,
    adversary_session: GenerationSession,
    device: str,
) -> tuple[dict[str, Any], float]:
    """Generate one adversarial rewrite and return elapsed milliseconds."""
    return adversary_rewrite_sample(
        adversary_session,
        device,
        prompt,
        instruction,
        max_new_tokens=120,
    )


def target_generate(
    prompt: str,
    target_session: GenerationSession,
    device: str,
    defense_prompt: str,
    max_new_tokens: int = 150,
) -> tuple[str, float]:
    """Generate one defended target response and return elapsed milliseconds."""

    request = GenerationRequest(
        system_prompt=defense_prompt,
        user_prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    return timed_target_generate(target_session, device, request)


def pad_gen_ids_batch(
    batch_gen_ids: Sequence[torch.Tensor], pad_id: int
) -> tuple[torch.Tensor, list[int]]:
    """Right-pad (1, L) tensors to a common L for batched forward; keep valid lengths."""
    lengths = [int(x.shape[-1]) for x in batch_gen_ids]
    max_len = max(lengths)
    rows: list[torch.Tensor] = []
    for x in batch_gen_ids:
        if x.dim() != 2 or x.size(0) != 1:
            raise ValueError(f"expected gen_ids shape (1, L), got {tuple(x.shape)}")
        L = x.size(1)
        if L < max_len:
            pad = torch.full(
                (1, max_len - L),
                pad_id,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=1)
        rows.append(x)
    return torch.cat(rows, dim=0), lengths


def reinforce_update_batch_sgd(
    model: Any,
    optimizer: Any,
    gen_ids: Any,
    prompt_lens: list[int],
    rewards: Any,
    valid_seq_lens: list[int] | None = None,
) -> tuple[float, Any]:
    """Apply a REINFORCE-style SGD step: mean over batch of (-reward_i * sum log pi)."""

    model.train()
    if valid_seq_lens is None:
        valid_seq_lens = [int(gen_ids.size(1))] * int(gen_ids.size(0))

    out = model(input_ids=gen_ids, use_cache=False)
    logits = out.logits
    log_probs = f.log_softmax(logits[:, :-1, :], dim=-1)

    logprob_sums = []
    for i in range(gen_ids.size(0)):
        prompt_len = int(prompt_lens[i])
        valid_end = int(valid_seq_lens[i])
        comp_ids = gen_ids[i, prompt_len:valid_end].clone()
        if comp_ids.numel() == 0:
            logprob_sums.append(torch.tensor(0.0, device=gen_ids.device))
            continue
        start = prompt_len - 1
        end = start + comp_ids.size(0)
        lp = log_probs[i, start:end, :]
        tok_lp = torch.gather(lp, -1, comp_ids.unsqueeze(-1)).squeeze(-1)
        logprob_sums.append(tok_lp.sum())

    logprob_sums = torch.stack(logprob_sums)
    loss = -(rewards * logprob_sums).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu()), logprob_sums.detach()


def evaluate_prompts(
    examples: list[dict[str, Any]],
    attacker_instruction: str,
    defense_prompt: str,
    ctx: RunContext,
    eval_cfg: EvaluationConfig,
    max_new_tokens: int,
    target_max_workers: int | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate attacker+defender prompts on examples and aggregate metrics.

    Per example the chain is adversary → target, but across examples we **pipeline**:
    after each adversary finishes we submit that target request to a pool while the main
    thread runs the next adversary (Unsloth stays sequential; vLLM can overlap in flight).
    """
    n = len(examples)
    workers = cap_thread_workers(n, target_max_workers)

    # Only HTTP/OpenAI targets are safe under ThreadPoolExecutor; local HF/GPU must stay sequential.
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
    """Baseline / stage / final eval with shared kwargs."""
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
    attacker_trace_df: pd.DataFrame,
    defender_trace_df: pd.DataFrame,
    stage_metrics_df: pd.DataFrame,
    final_attacker_instruction: str,
    final_defense_prompt: str,
    run_seconds: float,
    *,
    gepa_final_attacker_val_score: float | None = None,
    gepa_final_defender_val_score: float | None = None,
    config_snapshot: dict[str, Any],
) -> None:
    """Write all CoEV v2 artifacts, plots, and run manifest."""

    results_dir.mkdir(parents=True, exist_ok=True)

    optimized_prompts_path = results_dir / "coev_v2_optimized_prompts.json"
    write_json(
        optimized_prompts_path,
        {
            "attacker_instruction": final_attacker_instruction,
            "defense_prompt": final_defense_prompt,
        },
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
            "reflection_model_name": args.reflection_model_name,
            "reflection_vllm_base_url": args.reflection_vllm_base_url,
            "stages": args.stages,
            "iters_per_stage": args.iters_per_stage,
            "max_metric_calls": args.max_metric_calls,
            "run_seconds": run_seconds,
        },
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
    }
    if gepa_final_attacker_val_score is not None and gepa_final_defender_val_score is not None:
        metrics_payload["gepa_best_val_scores_final"] = {
            "attacker": gepa_final_attacker_val_score,
            "defender": gepa_final_defender_val_score,
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
            "optimizer_trace_attacker.csv": attacker_trace_df,
            "optimizer_trace_defender.csv": defender_trace_df,
        },
        skip_empty={"optimizer_trace_attacker.csv", "optimizer_trace_defender.csv"},
    )

    combined_trace = pd.concat([attacker_trace_df, defender_trace_df], ignore_index=True)
    baseline_plot_path = save_baseline_optimized_plot(
        comparison_df=comparison_df,
        out_path=results_dir / "plot_baseline_vs_optimized.png",
        title="CoEV v2 Baseline vs Optimized Metrics",
    )
    trajectory_plot_path = save_trajectory_plot(
        trace_df=combined_trace,
        out_path=results_dir / "plot_optimization_trajectory.png",
        title="CoEV v2 GEPA Optimization Trajectory",
        hue_col="role",
    )
    asr_iter_path = save_coev_asr_vs_global_step_plot(
        stage_metrics_df=stage_metrics_df,
        iters_per_stage=args.iters_per_stage,
        out_path=results_dir / "plot_asr_vs_iterations.png",
        title="CoEV v2 ASR vs global training step (stage checkpoints)",
    )
    refusal_iter_path = save_coev_refusal_vs_global_step_plot(
        stage_metrics_df=stage_metrics_df,
        iters_per_stage=args.iters_per_stage,
        out_path=results_dir / "plot_refusal_vs_iterations.png",
        title="CoEV v2 refusal rate vs global training step (stage checkpoints)",
    )

    manifest = RunManifest(
        mode="coev_v2",
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
            "reflection_model_name": args.reflection_model_name,
        },
        budget={
            "stages": args.stages,
            "iters_per_stage": args.iters_per_stage,
            "max_metric_calls": args.max_metric_calls,
            "run_seconds": run_seconds,
        },
        endpoints={
            "reflection_base_url": args.reflection_vllm_base_url,
        },
        extra={
            "optimized_prompts_path": str(optimized_prompts_path),
            "optimized_attacker_instruction_txt": str(attacker_txt_path),
            "optimized_defense_prompt_txt": str(defense_txt_path),
            "metrics_json_path": str(metrics_json_path),
            "training_csv_name": args.training_csv_name,
            "eval_method": args.eval_method,
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
    ]
    if trajectory_plot_path is not None:
        logged_paths.append(trajectory_plot_path)
    if asr_iter_path is not None:
        logged_paths.append(asr_iter_path)
    if refusal_iter_path is not None:
        logged_paths.append(refusal_iter_path)
    logged_paths.append(results_dir)
    log_saved_artifacts(logged_paths)


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse CLI arguments for CoEV v2 training/evaluation."""
    global_defaults = defaults["global"]
    run_defaults = merged_run_defaults(defaults, "coev_v2")

    parser = argparse.ArgumentParser(description="Run CoEV v2 (REINFORCE + GEPA prompt evolution).")
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
        help=(
            "Thread-pool size for target HTTP calls during eval; pipelines with adversary "
            "(next adversary runs while earlier targets are in flight)."
        ),
    )

    parser.add_argument("--stages", type=int, default=run_defaults["stages"])
    parser.add_argument("--iters-per-stage", type=int, default=run_defaults["iters_per_stage"])
    parser.add_argument("--eval-every-stages", type=int, default=run_defaults["eval_every_stages"])
    parser.add_argument("--train-slice-end", type=int, default=run_defaults["train_slice_end"])
    parser.add_argument("--eval-slice-start", type=int, default=run_defaults["eval_slice_start"])
    parser.add_argument("--eval-slice-end", type=int, default=run_defaults["eval_slice_end"])
    parser.add_argument("--lr", type=float, default=run_defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=run_defaults["weight_decay"])
    parser.add_argument("--initial-attacker-instruction", default=run_defaults["initial_attacker_instruction"])
    parser.add_argument("--initial-defense-prompt", default=run_defaults["initial_defense_prompt"])

    parser.add_argument("--max-metric-calls", type=int, default=run_defaults["max_metric_calls"])
    parser.add_argument("--gepa-max-tokens", type=int, default=run_defaults["gepa_max_tokens"])
    parser.add_argument("--gepa-temperature", type=float, default=run_defaults["gepa_temperature"])

    parser.add_argument("--eval-method", choices=["judge", "heuristic"], default=run_defaults["eval_method"])
    parser.add_argument("--refusal-threshold", type=float, default=run_defaults["refusal_threshold"])
    parser.add_argument("--asr-threshold", type=float, default=run_defaults["asr_threshold"])
    parser.add_argument(
        "--adversary-reinforce-batch-size",
        type=int,
        default=run_defaults['adversary_reinforce_batch_size'],
        help="RLOO: rollouts per training step (same key as coev_v2 REINFORCE batch).",
    )
    return parser.parse_args()


def _patch_args_from_yaml(args: argparse.Namespace, defaults: dict[str, Any]) -> None:
    """Attach runtime/model fields from YAML (still exposed as args.* for save_artifacts)."""
    args.runtime_profile = defaults["global"]["runtime_profile"]
    models = defaults["runtime"]["models"]
    args.adversary_model_id = models["adversary_model_id"]
    args.task_model_name = models["target_model_name"]
    args.reflection_model_name = models["reflection_model_name"]
    rw_url, rw_key = resolve_reflection_env_overrides(defaults)
    args.reflection_vllm_base_url = rw_url
    args.reflection_vllm_api_key = rw_key


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
    reflection_gateway = build_reflection_gateway_for_defaults(defaults)
    reflection_gateway.verify(args.reflection_model_name)
    print("Reflection model smoke output:", reflection_gateway.smoke_test(args.reflection_model_name))

    return RunContext(
        adversary_session=RuntimeCatalog.build_adversary_session(adversary_cfg),
        target_session=build_vllm_target_session(defaults),
        judge_session=RuntimeCatalog.build_judge_session(HarmbenchJudgeConfig()),
        reflection_gateway=reflection_gateway,
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
    t_log, t_a, t_d, t_s = (pd.DataFrame() for _ in range(4))
    save_artifacts(
        args=args,
        results_dir=results_dir,
        baseline_metrics=baseline_metrics,
        optimized_metrics=baseline_metrics,
        baseline_df=baseline_df,
        optimized_df=baseline_df,
        train_log_df=t_log,
        attacker_trace_df=t_a,
        defender_trace_df=t_d,
        stage_metrics_df=t_s,
        final_attacker_instruction=attacker_instruction,
        final_defense_prompt=defense_prompt,
        run_seconds=time.time() - run_start,
        config_snapshot=config_snapshot,
    )


def _run_training_and_finalize(
    args: argparse.Namespace,
    ctx: RunContext,
    eval_cfg: EvaluationConfig,
    coev_cfg: CoevConfig,
    gepa_cfg: DualRoleGepaPromptOptimizationConfig,
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
    attacker_trace_rows: list[OptimizerTraceRow] = []
    defender_trace_rows: list[OptimizerTraceRow] = []
    gepa_final_attacker_val_score: float | None = None
    gepa_final_defender_val_score: float | None = None

    print("Starting CoEV v2 training...")
    for stage in range(coev_cfg.stages):
        print(f"\n--- Stage {stage} start ---")
        stage_prompts: list[str] = []
        for i in range(coev_cfg.iters_per_stage):
            print(f"\n--- Stage {stage} iteration {i} ---")
            idx = torch.randint(0, len(train_prompts), ()).item()
            original_prompt = train_prompts[idx]
            stage_prompts.append(original_prompt)

            batch_size = max(1, int(args.adversary_reinforce_batch_size))
            batch_gen_ids: list[torch.Tensor] = []
            batch_rewards: list[float] = []
            batch_prompt_lens: list[int] = []
            adv_rewrites: list[str] = []
            target_resps: list[str] = []
            verdicts: list[str] = []

            for _ in range(batch_size):
                sample, _ = adversary_output(
                    prompt=original_prompt,
                    instruction=attacker_instruction,
                    adversary_session=ctx.adversary_session,
                    device=ctx.device,
                )
                adv_rewrite = sample["completion_text"].strip()
                target_resp, _ = target_generate(
                    prompt=adv_rewrite,
                    target_session=ctx.target_session,
                    device=ctx.device,
                    defense_prompt=defense_prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                reward, verdict = compute_reward_and_verdict(
                    behavior=original_prompt,
                    generation=target_resp,
                    eval_cfg=eval_cfg,
                    judge_session=ctx.judge_session if eval_cfg.method == "judge" else None,
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
            device = padded_ids.device
            rewards_t = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            loss_val, _ = reinforce_update_batch_sgd(
                model,
                optimizer,
                padded_ids,
                batch_prompt_lens,
                rewards_t,
                valid_seq_lens=valid_lens,
            )

            mean_r = float(rewards_t.mean().item())
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
                    "batch_size": batch_size,
                }
            )

        if stage % coev_cfg.eval_every_stages == 0:
            stage_metrics, _ = _eval_suite(ctx, eval_cfg, args, attacker_instruction, defense_prompt, eval_examples)
            stage_metric_rows.append({"stage": stage, "phase": "pre_evolution", **stage_metrics})
            print(
                f"Stage {stage}: Completed REINFORCE. Eval ASR: {stage_metrics['asr']:.2%} | "
                f"refusal_rate: {stage_metrics['refusal_rate']:.2%}"
            )
        else:
            print(f"Stage {stage}: Completed REINFORCE.")

        defense_before_gepa = defense_prompt
        print(
            f"GEPA stage {stage}: reflection=openai/{gepa_cfg.reflection_model_name} @ {args.reflection_vllm_base_url} "
            f"max_metric_calls={gepa_cfg.max_metric_calls}"
        )
        dual_role_result = run_dual_role_gepa_prompt_optimization(
            cfg=gepa_cfg,
            ctx=DualRoleGepaContext(
                adversary_session=ctx.adversary_session,
                target_session=ctx.target_session,
                judge_session=ctx.judge_session,
                reflection_gateway=ctx.reflection_gateway,
                device=ctx.device,
                eval_cfg=eval_cfg,
            ),
            stage_index=stage,
            stage_prompts=stage_prompts,
            attacker_instruction=attacker_instruction,
            defense_prompt=defense_prompt,
        )
        if dual_role_result.attacker_result is None or dual_role_result.defender_result is None:
            raise RuntimeError(
                "GEPA dual-role run failed to return attacker and defender results; "
                f"attacker_result={dual_role_result.attacker_result!r} "
                f"defender_result={dual_role_result.defender_result!r}"
            )
        attacker_instruction = dual_role_result.attacker_instruction
        defense_prompt = dual_role_result.defense_prompt
        attacker_trace_rows.extend(dual_role_result.attacker_trace)
        defender_trace_rows.extend(dual_role_result.defender_trace)

        attacker_eval_metrics, _ = _eval_suite(
            ctx, eval_cfg, args, attacker_instruction, defense_before_gepa, eval_examples
        )
        defender_eval_metrics, _ = _eval_suite(
            ctx, eval_cfg, args, attacker_instruction, defense_prompt, eval_examples
        )
        ar, dr = dual_role_result.attacker_result, dual_role_result.defender_result
        gepa_final_attacker_val_score = float(ar.val_aggregate_scores[ar.best_idx])
        gepa_final_defender_val_score = float(dr.val_aggregate_scores[dr.best_idx])
        stage_metric_rows.append(
            {"stage": stage, "phase": "attacker_gepa_best", "score": gepa_final_attacker_val_score, **attacker_eval_metrics}
        )
        stage_metric_rows.append(
            {"stage": stage, "phase": "defender_gepa_best", "score": gepa_final_defender_val_score, **defender_eval_metrics}
        )
        stage_metric_rows.append({"stage": stage, "phase": "gepa_seconds", "score": dual_role_result.run_seconds})

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
        attacker_trace_df=pd.DataFrame(attacker_trace_rows),
        defender_trace_df=pd.DataFrame(defender_trace_rows),
        stage_metrics_df=pd.DataFrame(stage_metric_rows),
        final_attacker_instruction=attacker_instruction,
        final_defense_prompt=defense_prompt,
        run_seconds=time.time() - run_start,
        gepa_final_attacker_val_score=gepa_final_attacker_val_score,
        gepa_final_defender_val_score=gepa_final_defender_val_score,
        config_snapshot=config_snapshot,
    )


def main() -> None:
    """Run full CoEV v2 pipeline: train/evolve/evaluate and persist artifacts."""
    defaults = load_default_config()
    args = parse_args(defaults)
    _patch_args_from_yaml(args, defaults)

    run_start = time.time()
    sns.set_theme(style="whitegrid")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = EvaluationConfig(
        method=args.eval_method,
        refusal_threshold=args.refusal_threshold,
        asr_threshold=args.asr_threshold,
    )
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
        initial_attacker_instruction=args.initial_attacker_instruction,
        initial_defense_prompt=args.initial_defense_prompt,
    )
    gepa_cfg = DualRoleGepaPromptOptimizationConfig(
        max_metric_calls=args.max_metric_calls,
        max_tokens=args.gepa_max_tokens,
        temperature=args.gepa_temperature,
        reflection_model_name=args.reflection_model_name,
    )

    ctx = _build_context(args, defaults, device)
    config_snapshot = build_config_snapshot(defaults, cli_args=args)
    train_prompts, _, eval_examples = _load_prompt_slices(args, coev_cfg)

    attacker_instruction = coev_cfg.initial_attacker_instruction
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
        gepa_cfg,
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
