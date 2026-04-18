#!/usr/bin/env python3
"""XSTest over-refusal evaluator.

Two modes:

* ``--mode target-only`` — run XSTest prompts directly against the target
  model using the configured system prompt. Measures baseline over-refusal of
  the defense.
* ``--mode adversary`` — load a trained adversary LoRA adapter from
  ``--checkpoint-dir``, rewrite each XSTest prompt through the adversary,
  then run the rewrite against the target. Measures whether the adversary
  causes the target to misclassify safe prompts (adversary-induced
  over-refusal or under-refusal).

Scoring uses :func:`src.evaluators.build_xstest_row` /
:func:`src.evaluators.summarize_xstest_metrics`. Artifacts:

* ``<results_dir>/xstest_results.csv`` — per-example rows
* ``<results_dir>/xstest_summary.json`` — aggregate metrics + config snapshot
* ``<results_dir>/run_manifest.json`` — manifest for the run

Config flow mirrors ``runs/adversary_run.py``: YAML defaults are loaded via
:func:`load_default_config`, merged with ``shared_generation``, then
argparse flags override. The runner reads its settings from
``runs.xstest`` in the active YAML.
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
import torch

from src.artifacts import log_saved_artifacts, write_json, write_many_csv, write_run_manifest
from src.dataset.xstest_loader import load_xstest_subset
from src.evaluators import build_xstest_row, summarize_xstest_metrics
from src.run_pipeline import adversary_rewrite_sample
from src.runtime import (
    GenerationRequest,
    GenerationSession,
    ModelConfig,
    RuntimeCatalog,
    UnslothAdversaryConfig,
    build_vllm_target_session,
    cap_thread_workers,
    resolve_hf_token,
    resolve_reflection_env_overrides,
    timed_target_generate,
)
from src.runtime.defaults import build_config_snapshot, load_default_config
from src.types import RunManifest


@dataclass
class RunContext:
    """Long-lived sessions used by one XSTest eval run."""

    target_session: GenerationSession
    adversary_session: GenerationSession | None
    device: str


def clean_text(value: Any) -> str:
    """Flatten newlines for single-line CSV rows."""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def _default_xstest_defaults() -> dict[str, Any]:
    """Fallback defaults when the YAML has no ``runs.xstest`` block.

    Kept permissive so the script runs against ``configs/default.yaml`` without
    requiring callers to edit it before first use.
    """
    return {
        "limit": None,
        "label_filter": None,
        "target_system_prompt": "You are a helpful assistant.",
        "attacker_instruction": None,
        "target_max_new_tokens": 256,
        "adversary_max_new_tokens": 256,
        "target_temperature": 0.0,
        "target_max_workers": 16,
        "results_dir": "results/xstest",
    }


def _resolve_xstest_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge ``runs.xstest`` with fallbacks so older YAML files still work."""
    out = _default_xstest_defaults()
    runs = defaults.get("runs", {})
    user = runs.get("xstest") if isinstance(runs, dict) else None
    if isinstance(user, dict):
        out.update({k: v for k, v in user.items() if v is not None})
    # Pull shared seed attacker instruction when user did not specify one.
    if out.get("attacker_instruction") is None:
        sg = defaults.get("shared_generation") or {}
        out["attacker_instruction"] = sg.get("seed_attacker_instruction")
    # target sampling defaults from shared_generation if runs.xstest did not override.
    sg = defaults.get("shared_generation") or {}
    samp = sg.get("sampling") or {}
    if samp.get("target_max_new_tokens") is not None and "target_max_new_tokens" not in (user or {}):
        out["target_max_new_tokens"] = samp["target_max_new_tokens"]
    if samp.get("adversary_max_new_tokens") is not None and "adversary_max_new_tokens" not in (user or {}):
        out["adversary_max_new_tokens"] = samp["adversary_max_new_tokens"]
    if samp.get("target_temperature") is not None and "target_temperature" not in (user or {}):
        out["target_temperature"] = samp["target_temperature"]
    if out.get("target_system_prompt") is None:
        out["target_system_prompt"] = sg.get("target_system_prompt", "You are a helpful assistant.")
    return out


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse CLI flags; defaults come from ``runs.xstest`` + ``shared_generation``."""
    global_defaults = defaults["global"]
    run_defaults = _resolve_xstest_defaults(defaults)

    parser = argparse.ArgumentParser(description="Run XSTest over-refusal evaluation.")
    parser.add_argument("--mode", choices=["target-only", "adversary"], default="target-only")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to a trained adversary LoRA checkpoint (required for --mode adversary).",
    )
    parser.add_argument("--device", default=global_defaults["device"])
    parser.add_argument("--seed", type=int, default=global_defaults["seed"])
    parser.add_argument("--results-dir", default=run_defaults["results_dir"])

    parser.add_argument(
        "--limit",
        type=int,
        default=run_defaults["limit"],
        help="Max XSTest examples to evaluate (default: all).",
    )
    parser.add_argument(
        "--label-filter",
        type=int,
        default=run_defaults["label_filter"],
        choices=[0, 1],
        help="0=safe, 1=unsafe; omit to evaluate both label classes.",
    )
    parser.add_argument(
        "--target-system-prompt",
        default=run_defaults["target_system_prompt"],
    )
    parser.add_argument(
        "--attacker-instruction",
        default=run_defaults["attacker_instruction"],
        help="Rewriter instruction; ignored in --mode target-only.",
    )
    parser.add_argument(
        "--target-max-new-tokens",
        type=int,
        default=run_defaults["target_max_new_tokens"],
    )
    parser.add_argument(
        "--adversary-max-new-tokens",
        type=int,
        default=run_defaults["adversary_max_new_tokens"],
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=run_defaults["target_temperature"],
    )
    parser.add_argument(
        "--target-max-workers",
        type=int,
        default=run_defaults["target_max_workers"],
        help="Thread-pool size for target HTTP completions when the backend is concurrency-safe.",
    )
    args = parser.parse_args()
    if args.mode == "adversary" and not args.checkpoint_dir:
        parser.error("--mode adversary requires --checkpoint-dir")
    if args.mode == "adversary" and not args.attacker_instruction:
        parser.error(
            "--mode adversary requires --attacker-instruction (or runs.xstest.attacker_instruction / "
            "shared_generation.seed_attacker_instruction in the active YAML)."
        )
    return args


def _build_target_context(args: argparse.Namespace, defaults: dict[str, Any]) -> RunContext:
    """Build a target-only run context (no adversary session)."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return RunContext(
        target_session=build_vllm_target_session(defaults),
        adversary_session=None,
        device=device,
    )


def _build_adversary_context(
    args: argparse.Namespace, defaults: dict[str, Any]
) -> RunContext:
    """Build an adversary+target run context and load adapters from ``--checkpoint-dir``."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig(model_id=defaults["runtime"]["models"]["adversary_model_id"])
    adversary_cfg = UnslothAdversaryConfig(
        model_id=model_cfg.model_id,
        max_seq_length=model_cfg.max_seq_length,
        load_in_4bit=model_cfg.load_in_4bit,
        lora_r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
    )
    adversary_session = RuntimeCatalog.build_adversary_session(adversary_cfg)

    checkpoint_path = Path(args.checkpoint_dir).resolve()
    print(f"Loading adversary adapter from: {checkpoint_path}")
    runtime = adversary_session.runtime
    if not hasattr(runtime, "load_adapters"):
        raise RuntimeError(
            "Adversary runtime does not expose load_adapters(); update src/runtime/local_runtimes.py."
        )
    runtime.load_adapters(checkpoint_path)

    return RunContext(
        target_session=build_vllm_target_session(defaults),
        adversary_session=adversary_session,
        device=device,
    )


def _run_target_only(
    examples: list[dict[str, Any]],
    ctx: RunContext,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run each XSTest prompt directly against the target with the defense system prompt."""
    n = len(examples)
    workers = cap_thread_workers(n, args.target_max_workers)
    use_pool = getattr(ctx.target_session.runtime, "supports_concurrent_target_inference", False)

    target_texts: list[str] = []
    target_latencies: list[float] = []

    if not use_pool:
        for ex in examples:
            request = GenerationRequest(
                system_prompt=args.target_system_prompt,
                user_prompt=ex["prompt"],
                max_new_tokens=args.target_max_new_tokens,
                temperature=args.target_temperature,
            )
            resp, latency_ms = timed_target_generate(ctx.target_session, ctx.device, request)
            target_texts.append(resp)
            target_latencies.append(latency_ms)
    else:
        pending: list[Future[tuple[str, float]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ex in examples:
                request = GenerationRequest(
                    system_prompt=args.target_system_prompt,
                    user_prompt=ex["prompt"],
                    max_new_tokens=args.target_max_new_tokens,
                    temperature=args.target_temperature,
                )
                pending.append(executor.submit(timed_target_generate, ctx.target_session, ctx.device, request))
            for fut in pending:
                resp, latency_ms = fut.result()
                target_texts.append(resp)
                target_latencies.append(latency_ms)

    rows: list[dict[str, Any]] = []
    for ex, target_resp, latency_ms in zip(examples, target_texts, target_latencies, strict=True):
        row = build_xstest_row(
            prompt=ex["prompt"],
            response=target_resp,
            is_safe=bool(ex["is_safe"]),
            category=str(ex.get("category", "unknown")),
        )
        row["id"] = str(ex["id"])
        row["adversary_prompt"] = ""
        row["target_latency_ms"] = float(latency_ms)
        row["adversary_latency_ms"] = 0.0
        rows.append(row)
    return rows


def _run_adversary_mode(
    examples: list[dict[str, Any]],
    ctx: RunContext,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Rewrite each XSTest prompt via the adversary, then evaluate target response."""
    if ctx.adversary_session is None:
        raise RuntimeError("Adversary mode requires an adversary_session in RunContext.")

    n = len(examples)
    workers = cap_thread_workers(n, args.target_max_workers)
    use_pool = getattr(ctx.target_session.runtime, "supports_concurrent_target_inference", False)

    adv_pairs: list[tuple[dict[str, Any], str, float]] = []
    target_texts: list[str] = []
    target_latencies: list[float] = []

    # Adversary sampling stays sequential (single GPU, 1 LoRA policy).
    # Target calls pipeline through a thread pool when the backend supports it.
    if not use_pool:
        for ex in examples:
            sample, adv_latency_ms = adversary_rewrite_sample(
                ctx.adversary_session,
                ctx.device,
                ex["prompt"],
                args.attacker_instruction,
                max_new_tokens=args.adversary_max_new_tokens,
            )
            adv_text = sample["completion_text"].strip()
            request = GenerationRequest(
                system_prompt=args.target_system_prompt,
                user_prompt=adv_text,
                max_new_tokens=args.target_max_new_tokens,
                temperature=args.target_temperature,
            )
            resp, target_latency_ms = timed_target_generate(ctx.target_session, ctx.device, request)
            adv_pairs.append((ex, adv_text, adv_latency_ms))
            target_texts.append(resp)
            target_latencies.append(target_latency_ms)
    else:
        pending: list[tuple[dict[str, Any], str, float, Future[tuple[str, float]]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ex in examples:
                sample, adv_latency_ms = adversary_rewrite_sample(
                    ctx.adversary_session,
                    ctx.device,
                    ex["prompt"],
                    args.attacker_instruction,
                    max_new_tokens=args.adversary_max_new_tokens,
                )
                adv_text = sample["completion_text"].strip()
                request = GenerationRequest(
                    system_prompt=args.target_system_prompt,
                    user_prompt=adv_text,
                    max_new_tokens=args.target_max_new_tokens,
                    temperature=args.target_temperature,
                )
                fut = executor.submit(timed_target_generate, ctx.target_session, ctx.device, request)
                pending.append((ex, adv_text, adv_latency_ms, fut))
            for ex, adv_text, adv_latency_ms, fut in pending:
                resp, target_latency_ms = fut.result()
                adv_pairs.append((ex, adv_text, adv_latency_ms))
                target_texts.append(resp)
                target_latencies.append(target_latency_ms)

    rows: list[dict[str, Any]] = []
    for (ex, adv_text, adv_latency_ms), target_resp, target_latency_ms in zip(
        adv_pairs, target_texts, target_latencies, strict=True
    ):
        row = build_xstest_row(
            prompt=ex["prompt"],
            response=target_resp,
            is_safe=bool(ex["is_safe"]),
            category=str(ex.get("category", "unknown")),
        )
        row["id"] = str(ex["id"])
        row["adversary_prompt"] = clean_text(adv_text)
        row["target_latency_ms"] = float(target_latency_ms)
        row["adversary_latency_ms"] = float(adv_latency_ms)
        rows.append(row)
    return rows


def main() -> None:
    """Run XSTest eval and persist artifacts."""
    run_start = time.time()
    defaults = load_default_config()
    args = parse_args(defaults)

    hf_token = resolve_hf_token()
    examples = load_xstest_subset(
        limit=args.limit,
        label_filter=args.label_filter,
        seed=args.seed,
        hf_token=hf_token,
    )
    # ``load_xstest_subset`` returns XSTestExampleRow (TypedDict); cast to dict for row-level use.
    examples = [dict(ex) for ex in examples]
    print(f"Loaded {len(examples)} XSTest examples (label_filter={args.label_filter}, limit={args.limit})")

    if args.mode == "adversary":
        ctx = _build_adversary_context(args, defaults)
        rows = _run_adversary_mode(examples, ctx, args)
    else:
        ctx = _build_target_context(args, defaults)
        rows = _run_target_only(examples, ctx, args)

    metrics = summarize_xstest_metrics(rows)
    print("\nXSTest metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    rows_df = pd.DataFrame(rows)
    write_many_csv(results_dir, {"xstest_results.csv": rows_df})

    run_seconds = time.time() - run_start
    base_url, _api_key = resolve_reflection_env_overrides(defaults)
    summary_payload = {
        "config": {
            "mode": args.mode,
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()) if args.checkpoint_dir else None,
            "limit": args.limit,
            "label_filter": args.label_filter,
            "target_system_prompt": args.target_system_prompt,
            "attacker_instruction": args.attacker_instruction,
            "target_max_new_tokens": args.target_max_new_tokens,
            "adversary_max_new_tokens": args.adversary_max_new_tokens,
            "target_temperature": args.target_temperature,
            "reflection_vllm_base_url": base_url,
            "target_model_name": defaults["runtime"]["models"]["target_model_name"],
            "adversary_model_id": defaults["runtime"]["models"]["adversary_model_id"],
            "run_seconds": run_seconds,
        },
        "metrics": metrics,
        "num_examples": len(rows),
    }
    summary_path = results_dir / "xstest_summary.json"
    write_json(summary_path, summary_payload)

    manifest = RunManifest(
        mode=f"xstest_{args.mode}",
        runtime_profile=defaults["global"].get("runtime_profile", "unknown"),
        seed=args.seed,
        dataset={"dataset_name": "Paul/XSTest", "limit": args.limit, "label_filter": args.label_filter},
        models={
            "target_model": defaults["runtime"]["models"]["target_model_name"],
            "adversary_model": (
                defaults["runtime"]["models"]["adversary_model_id"] if args.mode == "adversary" else None
            ),
            "judge_model": None,
        },
        budget={"num_examples": len(rows), "run_seconds": run_seconds},
        endpoints={"reflection_base_url": base_url},
        extra={
            "summary_path": str(summary_path),
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()) if args.checkpoint_dir else None,
            "mode": args.mode,
        },
        config_snapshot=build_config_snapshot(defaults, cli_args=args),
    )
    manifest_path = write_run_manifest(results_dir=results_dir, payload=manifest)
    log_saved_artifacts([results_dir / "xstest_results.csv", summary_path, manifest_path, results_dir])


if __name__ == "__main__":
    main()
