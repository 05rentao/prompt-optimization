#!/usr/bin/env python3
"""Unified runner for CoEV and Mark GEPA experiments.

This script keeps existing experiment entrypoints intact and orchestrates them
through one CLI surface:
- mark   -> runs/gepa_run.py
- coev   -> runs/coev_run.py
- coev_v2 -> runs/coev_v2_run.py
- hybrid -> runs both sequentially (order configurable)
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MARK_SCRIPT = REPO_ROOT / "runs" / "gepa_run.py"
COEV_SCRIPT = REPO_ROOT / "runs" / "coev_run.py"
COEV_V2_SCRIPT = REPO_ROOT / "runs" / "coev_v2_run.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CoEV/Mark experiment runner.")
    parser.add_argument("--mode", choices=["mark", "coev", "coev_v2", "hybrid"], required=True)
    parser.add_argument(
        "--hybrid-order",
        choices=["mark_then_coev", "coev_then_mark"],
        default="mark_then_coev",
        help="Execution order when mode=hybrid.",
    )
    parser.add_argument("--runtime-profile", default="dual_vllm")

    # Shared dataset defaults.
    parser.add_argument("--dataset-name", default="walledai/HarmBench")
    parser.add_argument("--dataset-config", default="standard")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Mark-specific defaults.
    parser.add_argument("--task-model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--reflection-model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--reflection-vllm-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--reflection-vllm-api-key", default="EMPTY")
    parser.add_argument("--max-metric-calls", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mark-results-dir", default="results/mark")
    parser.add_argument("--mark-extra-args", default="")

    # CoEV-specific defaults.
    parser.add_argument("--coev-mode", choices=["reinforce", "gepa", "eval"], default="reinforce")
    parser.add_argument("--device", default=None)
    parser.add_argument("--coev-results-dir", default="results/coev")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--eval-instruction", default=None)
    parser.add_argument("--coev-extra-args", default="")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build_mark_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(MARK_SCRIPT),
        "--dataset-name",
        args.dataset_name,
        "--dataset-config",
        args.dataset_config,
        "--dataset-split",
        args.dataset_split,
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--seed",
        str(args.seed),
        "--task-model-name",
        args.task_model_name,
        "--reflection-model-name",
        args.reflection_model_name,
        "--vllm-base-url",
        args.vllm_base_url,
        "--vllm-api-key",
        args.vllm_api_key,
        "--reflection-vllm-base-url",
        args.reflection_vllm_base_url,
        "--reflection-vllm-api-key",
        args.reflection_vllm_api_key,
        "--max-metric-calls",
        str(args.max_metric_calls),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--runtime-profile",
        args.runtime_profile,
        "--results-dir",
        args.mark_results_dir,
        "--show-progress",
    ]
    if args.mark_extra_args.strip():
        cmd.extend(shlex.split(args.mark_extra_args))
    return cmd


def build_coev_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(COEV_SCRIPT),
        "--mode",
        args.coev_mode,
        "--dataset-name",
        args.dataset_name,
        "--dataset-config",
        args.dataset_config,
        "--dataset-split",
        args.dataset_split,
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--seed",
        str(args.seed),
        "--runtime-profile",
        args.runtime_profile,
        "--results-dir",
        args.coev_results_dir,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.save_dir:
        cmd.extend(["--save-dir", args.save_dir])
    if args.eval_instruction:
        cmd.extend(["--eval-instruction", args.eval_instruction])
    if args.coev_extra_args.strip():
        cmd.extend(shlex.split(args.coev_extra_args))
    return cmd


def build_coev_v2_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(COEV_V2_SCRIPT),
        "--mode",
        "coev" if args.coev_mode in {"reinforce", "gepa"} else "eval",
        "--dataset-name",
        args.dataset_name,
        "--dataset-config",
        args.dataset_config,
        "--dataset-split",
        args.dataset_split,
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--seed",
        str(args.seed),
        "--runtime-profile",
        args.runtime_profile,
        "--results-dir",
        args.coev_results_dir,
        "--reflection-model-name",
        args.reflection_model_name,
        "--reflection-vllm-base-url",
        args.reflection_vllm_base_url,
        "--reflection-vllm-api-key",
        args.reflection_vllm_api_key,
        "--max-metric-calls",
        str(args.max_metric_calls),
        "--max-new-tokens",
        str(args.max_tokens),
        "--gepa-max-tokens",
        str(args.max_tokens),
        "--gepa-temperature",
        str(args.temperature),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.save_dir:
        cmd.extend(["--save-dir", args.save_dir])
    if args.coev_extra_args.strip():
        cmd.extend(shlex.split(args.coev_extra_args))
    return cmd


def main() -> None:
    args = parse_args()

    if args.mode == "mark":
        run_command(build_mark_command(args))
        return

    if args.mode == "coev":
        run_command(build_coev_command(args))
        return

    if args.mode == "coev_v2":
        run_command(build_coev_v2_command(args))
        return

    # mode == "hybrid"
    if args.hybrid_order == "mark_then_coev":
        run_command(build_mark_command(args))
        run_command(build_coev_command(args))
    else:
        run_command(build_coev_command(args))
        run_command(build_mark_command(args))


if __name__ == "__main__":
    main()
