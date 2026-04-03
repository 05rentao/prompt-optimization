#!/usr/bin/env python3
"""GEPA prompt optimization script converted from mark_exp.ipynb.

This script runs an end-to-end prompt optimization cycle:
1) Load a HarmBench subset from Hugging Face.
2) Evaluate a baseline safety prompt on harmful requests.
3) Optimize the system prompt with GEPA.
4) Re-evaluate optimized prompt and export artifacts.

Target generation uses the same OpenAI-compatible vLLM server as GEPA reflection
(see ``configs/default.yaml`` ``runtime.reflection`` and ``runtime.models``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path when invoked as `python runs/gepa_run.py` (default sys.path is `runs/`).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import random
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.auto import tqdm

from src.artifacts import (
    log_saved_artifacts,
    save_baseline_optimized_plot,
    save_gepa_refusal_vs_evaluator_calls_plot,
    save_trajectory_plot,
    write_json,
    write_many_csv,
    write_run_manifest,
    write_text,
)
from src.data import load_harmbench_subset
from src.evaluators import compute_refusal_score
from src.runtime import (
    EvaluationConfig,
    GenerationRequest,
    GenerationSession,
    GepaPromptOptimizationConfig,
    HarmbenchJudgeConfig,
    OpenAIReflectionGateway,
    RuntimeCatalog,
    build_vllm_stack,
    cap_thread_workers,
    evaluate_outputs,
    patch_run_args_from_config,
    run_gepa_prompt_optimization,
    resolve_hf_token,
    timed_target_generate,
)
from src.runtime.defaults import build_config_snapshot, load_default_config
from src.types import RunManifest


@dataclass
class EvalContext:
    """Target + judge sessions for GEPA eval (no adversary in this pipeline)."""

    target_session: GenerationSession
    judge_session: GenerationSession | None
    device: str


def evaluate_system_prompt(
    system_prompt: str,
    examples: list[dict[str, Any]],
    ctx: EvalContext,
    eval_cfg: EvaluationConfig,
    max_tokens: int,
    temperature: float,
    target_max_workers: int | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate one system prompt over examples and aggregate refusal/ASR metrics.

    When the target runtime supports concurrent HTTP inference, completions are run
    in a thread pool; otherwise examples are processed sequentially. Progress uses tqdm.
    """
    n = len(examples)
    workers = cap_thread_workers(n, target_max_workers)

    use_target_pool = getattr(ctx.target_session.runtime, "supports_concurrent_target_inference", False)

    results: list[tuple[dict[str, Any], str, float]] = []

    if not use_target_pool:
        iterable: Any = tqdm(examples, desc="Evaluating")
        for ex in iterable:
            request = GenerationRequest(
                system_prompt=system_prompt,
                user_prompt=ex["prompt"],
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            output, elapsed_ms = timed_target_generate(ctx.target_session, ctx.device, request)
            results.append((ex, output, elapsed_ms))
    else:
        pending: list[tuple[dict[str, Any], Future[tuple[str, float]]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ex in examples:
                request = GenerationRequest(
                    system_prompt=system_prompt,
                    user_prompt=ex["prompt"],
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                fut = executor.submit(timed_target_generate, ctx.target_session, ctx.device, request)
                pending.append((ex, fut))

            iterator = tqdm(pending, desc="Evaluating")
            for ex, fut in iterator:
                output, elapsed_ms = fut.result()
                results.append((ex, output, elapsed_ms))

    behaviors: list[str] = []
    generations: list[str] = []
    rows: list[dict[str, Any]] = []
    for ex, output, elapsed_ms in results:
        r_score = compute_refusal_score(output)
        behaviors.append(ex["prompt"])
        generations.append(output)
        rows.append(
            {
                "id": str(ex["id"]),
                "behavior": ex["prompt"],
                "generation": output,
                "prompt": ex["prompt"],
                "response": output,
                "refusal_score": r_score,
                "asr_score": 1.0 - r_score,
                "latency_ms": elapsed_ms,
            }
        )

    eval_result = evaluate_outputs(
        behaviors=behaviors,
        generations=generations,
        cfg=eval_cfg,
        judge_session=ctx.judge_session,
    )
    aggregate_score = eval_result.mean_refusal_score
    if aggregate_score is None:
        aggregate_score = eval_result.refusal_rate

    df = pd.DataFrame(rows)
    metrics = {
        "refusal_rate": float(eval_result.refusal_rate),
        "asr": float(eval_result.asr),
        "mean_refusal_score": float(df["refusal_score"].mean()) if not df.empty else 0.0,
        "aggregate_score": float(aggregate_score),
        "latency_ms_mean": float(df["latency_ms"].mean()) if not df.empty else 0.0,
    }
    return metrics, df


def _eval_suite(
    system_prompt: str,
    examples: list[dict[str, Any]],
    ctx: EvalContext,
    eval_cfg: EvaluationConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Baseline / final eval with shared kwargs."""
    return evaluate_system_prompt(
        system_prompt=system_prompt,
        examples=examples,
        ctx=ctx,
        eval_cfg=eval_cfg,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        target_max_workers=args.target_max_workers,
    )


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse CLI arguments for the GEPA optimization pipeline."""
    global_defaults = defaults["global"]
    run_defaults = defaults["runs"]["gepa"]
    parser = argparse.ArgumentParser(description="Run GEPA optimization for HarmBench safety prompt tuning.")
    parser.add_argument("--dataset-name", default=global_defaults["dataset_name"])
    parser.add_argument("--dataset-config", default=global_defaults["dataset_config"])
    parser.add_argument("--dataset-split", default=global_defaults["dataset_split"])
    parser.add_argument("--train-size", type=int, default=run_defaults["train_size"])
    parser.add_argument("--val-size", type=int, default=run_defaults["val_size"])
    parser.add_argument("--seed", type=int, default=global_defaults["seed"])

    parser.add_argument("--device", default=global_defaults["device"], help="Device override (e.g. cuda, cpu).")

    parser.add_argument("--max-metric-calls", type=int, default=run_defaults["max_metric_calls"])
    parser.add_argument("--max-tokens", type=int, default=run_defaults["max_tokens"])
    parser.add_argument("--temperature", type=float, default=run_defaults["temperature"])
    parser.add_argument(
        "--target-max-workers",
        type=int,
        default=run_defaults.get("target_max_workers", 16),
        help=(
            "Thread-pool size for concurrent target HTTP completions during eval "
            "(when the target runtime supports it)."
        ),
    )
    parser.add_argument("--eval-method", choices=["heuristic", "judge"], default=run_defaults["eval_method"])
    parser.add_argument("--refusal-threshold", type=float, default=run_defaults["refusal_threshold"])
    parser.add_argument("--asr-threshold", type=float, default=run_defaults["asr_threshold"])

    parser.add_argument("--baseline-system-prompt", default=run_defaults["baseline_system_prompt"])
    parser.add_argument("--results-dir", default=run_defaults["results_dir"])
    return parser.parse_args()


def _judge_session_for_gepa_eval(args: argparse.Namespace, defaults: dict[str, Any]) -> GenerationSession | None:
    """HarmBench judge session when ``eval_method == judge``."""
    if args.eval_method != "judge":
        return None
    judge_model_id = defaults["runtime"]["models"]["judge_model_id"]
    return RuntimeCatalog.build_judge_session(HarmbenchJudgeConfig(model_id=judge_model_id))


def verify_reflection_client(reflection_gateway: OpenAIReflectionGateway, reflection_model_name: str) -> None:
    """Validate reflection endpoint reachability and run a smoke prompt."""
    reflection_gateway.verify(reflection_model_name)
    reflection_smoke = reflection_gateway.smoke_test(reflection_model_name)
    print("Reflection model smoke output:", reflection_smoke)


def extract_best_candidate_and_score(result_obj: Any) -> tuple[dict[str, Any], float | None]:
    """Extract best GEPA candidate payload and its validation score."""
    candidate = result_obj.best_candidate
    score = result_obj.val_aggregate_scores[result_obj.best_idx]
    return candidate, score


def run_gepa_optimization(
    args: argparse.Namespace,
    target_session: GenerationSession,
    reflection_gateway: OpenAIReflectionGateway,
    device: str,
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
    baseline_system_prompt: str,
    eval_cfg: EvaluationConfig,
    judge_session: GenerationSession | None,
) -> tuple[Any, list[dict[str, Any]], float]:
    """Run GEPA optimization loop and collect per-call trace metadata."""
    optimization_cfg = GepaPromptOptimizationConfig(
        max_metric_calls=args.max_metric_calls,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reflection_model_name=args.reflection_model_name,
    )

    print("Starting GEPA optimization...")
    print(f"GEPA inner-loop eval: {eval_cfg.method}")
    print(f"Target model:      {args.target_model_name}")
    print(f"Reflection model:  openai/{args.reflection_model_name}")
    print(f"Reflection URL:    {args.reflection_vllm_base_url}")
    print(f"Budget:            {args.max_metric_calls} evaluator calls")
    result, optimizer_trace, run_seconds = run_gepa_prompt_optimization(
        cfg=optimization_cfg,
        target_session=target_session,
        reflection_gateway=reflection_gateway,
        device=device,
        train_data=train_data,
        val_data=val_data,
        baseline_system_prompt=baseline_system_prompt,
        eval_cfg=eval_cfg,
        judge_session=judge_session,
    )
    print(f"GEPA optimization finished in {run_seconds:.1f} seconds.")
    return result, optimizer_trace, run_seconds


def save_artifacts(
    results_dir: Path,
    args: argparse.Namespace,
    max_metric_calls: int,
    run_seconds: float,
    baseline_metrics: dict[str, float],
    optimized_metrics: dict[str, float],
    best_score: float | None,
    comparison_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    optimizer_trace: list[dict[str, Any]],
    optimized_system_prompt: str,
    config_snapshot: dict[str, Any],
) -> None:
    """Persist prompt, metrics, tables, plots, and run manifest artifacts."""
    optimized_prompt_path = results_dir / "optimized_system_prompt.txt"
    write_text(optimized_prompt_path, optimized_system_prompt)

    metrics_payload = {
        "config": {
            "target_model_name": args.target_model_name,
            "reflection_model_name": args.reflection_model_name,
            "reflection_vllm_base_url": args.reflection_vllm_base_url,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "max_metric_calls": max_metric_calls,
            "run_seconds": run_seconds,
            "runtime_profile": args.runtime_profile,
        },
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
        "best_score_from_gepa": best_score,
    }
    metrics_json_path = results_dir / "gepa_run_metrics.json"
    write_json(metrics_json_path, metrics_payload)

    manifest = RunManifest(
        mode="gepa",
        runtime_profile=args.runtime_profile,
        seed=args.seed,
        dataset={
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        models={
            "target_model_name": args.target_model_name,
            "reflection_model_name": args.reflection_model_name,
        },
        budget={
            "max_metric_calls": max_metric_calls,
            "run_seconds": run_seconds,
        },
        endpoints={
            "reflection_base_url": args.reflection_vllm_base_url,
        },
        extra={
            "best_score_from_gepa": best_score,
            "optimized_prompt_path": str(optimized_prompt_path),
            "metrics_json_path": str(metrics_json_path),
        },
        config_snapshot=config_snapshot,
    )
    manifest_path = write_run_manifest(results_dir=results_dir, payload=manifest)

    csv_outputs = write_many_csv(
        results_dir,
        {
            "baseline_vs_optimized_metrics.csv": comparison_df,
            "baseline_eval_outputs.csv": baseline_df,
            "optimized_eval_outputs.csv": optimized_df,
        },
    )
    comparison_csv_path = csv_outputs["baseline_vs_optimized_metrics.csv"]

    trace_df = pd.DataFrame(optimizer_trace)
    if not trace_df.empty:
        write_many_csv(results_dir, {"optimizer_trace.csv": trace_df})

    fig1_path = save_baseline_optimized_plot(
        comparison_df=comparison_df,
        out_path=results_dir / "plot_baseline_vs_optimized.png",
        title="Baseline vs Optimized Metrics",
    )

    fig2_path = results_dir / "plot_optimization_trajectory.png"
    trajectory_path = save_trajectory_plot(
        trace_df=trace_df,
        out_path=fig2_path,
        title="GEPA Optimization Trajectory",
    )
    refusal_calls_path = save_gepa_refusal_vs_evaluator_calls_plot(
        trace_df=trace_df,
        out_path=results_dir / "plot_refusal_vs_evaluator_calls.png",
        title="GEPA refusal vs evaluator call",
    )

    logged_paths = [optimized_prompt_path, metrics_json_path, comparison_csv_path, manifest_path, fig1_path]
    if trajectory_path is not None:
        logged_paths.append(trajectory_path)
    if refusal_calls_path is not None:
        logged_paths.append(refusal_calls_path)
    logged_paths.append(results_dir)
    log_saved_artifacts(logged_paths)


def main() -> None:
    """Run full GEPA pipeline: load data, optimize, evaluate, and save outputs."""
    defaults = load_default_config()
    args = parse_args(defaults)
    patch_run_args_from_config(defaults, args, run="gepa")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = EvaluationConfig(
        method=args.eval_method,
        refusal_threshold=args.refusal_threshold,
        asr_threshold=args.asr_threshold,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    sns.set_theme(style="whitegrid")

    target_session, reflection_gateway = build_vllm_stack(defaults)
    ctx = EvalContext(
        target_session=target_session,
        judge_session=_judge_session_for_gepa_eval(args, defaults),
        device=device,
    )
    verify_reflection_client(reflection_gateway, args.reflection_model_name)

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
    print(f"Loaded train={len(train_data)}, val={len(val_data)} from {args.dataset_name}:{args.dataset_split}")

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics, baseline_df = _eval_suite(
        args.baseline_system_prompt,
        val_data,
        ctx,
        eval_cfg,
        args,
    )
    print("Baseline metrics:")
    for key, value in baseline_metrics.items():
        print(f"  {key}: {value:.4f}")

    gepa_result, optimizer_trace, run_seconds = run_gepa_optimization(
        args=args,
        target_session=ctx.target_session,
        reflection_gateway=reflection_gateway,
        device=device,
        train_data=train_data,
        val_data=val_data,
        baseline_system_prompt=args.baseline_system_prompt,
        eval_cfg=eval_cfg,
        judge_session=ctx.judge_session,
    )
    best_candidate, best_score = extract_best_candidate_and_score(gepa_result)
    optimized_system_prompt = best_candidate.get("system_prompt", args.baseline_system_prompt)
    print("Best score from GEPA:", best_score)

    optimized_metrics, optimized_df = _eval_suite(
        optimized_system_prompt,
        val_data,
        ctx,
        eval_cfg,
        args,
    )
    print("Optimized metrics:")
    for key, value in optimized_metrics.items():
        print(f"  {key}: {value:.4f}")

    comparison_df = pd.DataFrame(
        [
            {"variant": "baseline", **baseline_metrics},
            {"variant": "optimized", **optimized_metrics},
        ]
    )

    save_artifacts(
        results_dir=results_dir,
        args=args,
        max_metric_calls=args.max_metric_calls,
        run_seconds=run_seconds,
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        best_score=best_score,
        comparison_df=comparison_df,
        baseline_df=baseline_df,
        optimized_df=optimized_df,
        optimizer_trace=optimizer_trace,
        optimized_system_prompt=optimized_system_prompt,
        config_snapshot=build_config_snapshot(defaults, cli_args=args),
    )


if __name__ == "__main__":
    main()
