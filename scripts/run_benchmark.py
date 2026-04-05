#!/usr/bin/env python3
"""Universal benchmark runner for XSTest and HarmBench."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time
from datetime import datetime
from typing import Any

import torch
from openai import OpenAI

from src.dataset.xstest_loader import load_xstest_subset
from src.data import load_harmbench_subset
from src.evaluators import (
    classify_xstest_response,
    summarize_xstest_metrics,
)
from src.runtime import (
    EvaluationConfig,
    GenerationRequest,
    GenerationSession,
    HarmbenchJudgeConfig,
    RuntimeCatalog,
    build_vllm_target_session,
    resolve_hf_token,
)
from src.runtime.defaults import load_default_config


def _benchmark_target_sampling(defaults: dict[str, Any]) -> tuple[str, int, float]:
    """System prompt and decoding from merged config (shared_generation + runs.adversary)."""
    adv = defaults.get("runs", {}).get("adversary", {})
    sg = defaults.get("shared_generation") or {}
    samp = sg.get("sampling") or {}
    system = adv.get("target_system_prompt") or sg.get("target_system_prompt", "You are a helpful assistant.")
    max_tok = int(adv.get("max_new_tokens") or samp.get("target_max_new_tokens") or 256)
    temp = float(samp.get("target_temperature", 0.0))
    return system, max_tok, temp


def run_xstest_benchmark(
    model_session: GenerationSession,
    examples: list[dict[str, Any]],
    device: str,
    output_dir: Path,
    *,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Run XSTest benchmark and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, example in enumerate(examples):
        request = GenerationRequest(
            system_prompt=system_prompt,
            user_prompt=example["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        response = model_session.generate(request, device=device)

        classification = classify_xstest_response(response)
        results.append({
            "id": example["id"],
            "prompt": example["prompt"],
            "response": response,
            "is_safe": example["is_safe"],
            "category": example.get("category", "unknown"),
            "classification": classification,
            "is_compliant": classification in ["full_compliance", "partial_compliance"],
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    # Save raw results
    results_file = output_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute metrics
    metrics = summarize_xstest_metrics(results)
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nXSTest Results:")
    print(f"  Safe compliance: {metrics['safe_compliance']:.1%}")
    print(f"  Unsafe ASR: {metrics['unsafe_asr']:.1%}")
    print(f"  Overrefusal rate: {metrics['overrefusal_rate']:.1%}")
    print(f"  Underrefusal rate: {metrics['underrefusal_rate']:.1%}")
    print(f"  Saved to: {output_dir}")

    return metrics


def run_harmbench_benchmark(
    model_session: GenerationSession,
    judge_session: GenerationSession,
    examples: list[dict[str, Any]],
    device: str,
    output_dir: Path,
    *,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Run HarmBench benchmark and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, example in enumerate(examples):
        request = GenerationRequest(
            system_prompt=system_prompt,
            user_prompt=example["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        response = model_session.generate(request, device=device)

        # Judge the response
        verdicts = judge_session.judge(
            behaviors=[example["prompt"]],
            generations=[response],
        )
        verdict = verdicts[0] if verdicts else "unknown"

        results.append({
            "id": example["id"],
            "prompt": example["prompt"],
            "response": response,
            "verdict": verdict,
            "asr_score": 1.0 if verdict.lower().startswith("yes") else 0.0,
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    # Save results
    results_file = output_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute metrics
    asr = sum(r["asr_score"] for r in results) / len(results) if results else 0.0
    metrics = {
        "num_examples": len(results),
        "asr": asr,
    }
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nHarmBench Results:")
    print(f"  ASR: {asr:.1%}")
    print(f"  Saved to: {output_dir}")

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Universal benchmark runner for XSTest and HarmBench")
    parser.add_argument(
        "--benchmark",
        choices=["xstest", "harmbench", "both"],
        default="both",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model ID (for vLLM/HF backends)",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf", "openai", "litellm"],
        default="vllm",
        help="Backend to use for inference",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="Base URL for OpenAI-compatible backend",
    )
    parser.add_argument(
        "--api-key",
        default="fake-key",
        help="API key for OpenAI-compatible backend",
    )
    parser.add_argument(
        "--xstest-limit",
        type=int,
        default=None,
        help="Max XSTest examples to run",
    )
    parser.add_argument(
        "--harmbench-limit",
        type=int,
        default=None,
        help="Max HarmBench examples to run",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/benchmark/<benchmark>_<model>_<timestamp>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to evaluate (overrides config)",
    )
    return parser.parse_args()


def main() -> None:
    """Run benchmarks."""
    args = parse_args()
    defaults = load_default_config()
    device = args.device
    hf_token = resolve_hf_token()
    sys_prompt, bench_max_tokens, bench_temp = _benchmark_target_sampling(defaults)
    if args.system_prompt:
        sys_prompt = args.system_prompt

    # Set up output directory
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path("results") / "benchmark" / f"{args.benchmark}_{args.model}_{timestamp}"

    # Load model session (vLLM backend via OpenAI-compatible API)
    if args.backend == "vllm":
        model_session = build_vllm_target_session(defaults)
    else:
        raise NotImplementedError(f"Backend {args.backend} not yet implemented")

    # Load judge for HarmBench
    judge_session = RuntimeCatalog.build_judge_session(HarmbenchJudgeConfig())

    # Run benchmarks
    if args.benchmark in ["xstest", "both"]:
        print("Running XSTest benchmark...")
        xstest_examples = load_xstest_subset(limit=args.xstest_limit, hf_token=hf_token)
        xstest_output = output_root / "xstest"
        run_xstest_benchmark(
            model_session,
            xstest_examples,
            device,
            xstest_output,
            system_prompt=sys_prompt,
            max_new_tokens=bench_max_tokens,
            temperature=bench_temp,
        )

    if args.benchmark in ["harmbench", "both"]:
        print("Running HarmBench benchmark...")
        harmbench_data, _, _ = load_harmbench_subset(
            dataset_name="walledai/HarmBench",
            dataset_config="standard",
            split="train",
            train_size=args.harmbench_limit or 100,
            val_size=0,
            seed=args.seed,
            hf_token=hf_token,
        )
        harmbench_output = output_root / "harmbench"
        run_harmbench_benchmark(
            model_session,
            judge_session,
            harmbench_data,
            device,
            harmbench_output,
            system_prompt=sys_prompt,
            max_new_tokens=bench_max_tokens,
            temperature=bench_temp,
        )

    print(f"\nAll benchmarks completed. Results saved to: {output_root}")


if __name__ == "__main__":
    main()
