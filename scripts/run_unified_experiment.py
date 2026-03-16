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

from src.runtime.defaults import load_default_config


REPO_ROOT = Path(__file__).resolve().parent.parent
MARK_SCRIPT = REPO_ROOT / "runs" / "gepa_run.py"
COEV_SCRIPT = REPO_ROOT / "runs" / "coev_run.py"
COEV_V2_SCRIPT = REPO_ROOT / "runs" / "coev_v2_run.py"


def parse_args() -> argparse.Namespace:
    defaults = load_default_config()
    global_defaults = defaults["global"]
    run_defaults = defaults["runs"]
    unified_defaults = defaults["scripts"]["unified_runner"]

    parser = argparse.ArgumentParser(description="Unified CoEV/Mark experiment runner.")
    parser.add_argument("--mode", choices=["mark", "coev", "coev_v2", "hybrid"], required=True)
    parser.add_argument(
        "--hybrid-order",
        choices=["mark_then_coev", "coev_then_mark"],
        default="mark_then_coev",
        help="Execution order when mode=hybrid.",
    )

    # Shared dataset defaults.
    parser.add_argument("--dataset-name", default=global_defaults["dataset_name"])
    parser.add_argument("--dataset-config", default=global_defaults["dataset_config"])
    parser.add_argument("--dataset-split", default=global_defaults["dataset_split"])
    parser.add_argument("--train-size", type=int, default=run_defaults["gepa"]["train_size"])
    parser.add_argument("--val-size", type=int, default=run_defaults["gepa"]["val_size"])
    parser.add_argument("--seed", type=int, default=global_defaults["seed"])

    # Mark-specific defaults.
    parser.add_argument("--max-metric-calls", type=int, default=run_defaults["gepa"]["max_metric_calls"])
    parser.add_argument("--max-tokens", type=int, default=run_defaults["gepa"]["max_tokens"])
    parser.add_argument("--temperature", type=float, default=run_defaults["gepa"]["temperature"])
    parser.add_argument("--mark-results-dir", default=unified_defaults["mark_results_dir"])
    parser.add_argument("--mark-extra-args", default="")

    # CoEV-specific defaults.
    parser.add_argument("--coev-mode", choices=["reinforce", "gepa", "eval"], default="reinforce")
    parser.add_argument("--coev-v2-mode", choices=["coev", "eval"], default=None)
    parser.add_argument("--device", default=global_defaults["device"])
    parser.add_argument("--coev-results-dir", default=unified_defaults["coev_results_dir"])
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--eval-instruction", default=None)
    parser.add_argument("--eval-method", choices=["judge", "heuristic"], default=run_defaults["coev"]["eval_method"])
    parser.add_argument("--refusal-threshold", type=float, default=run_defaults["coev"]["refusal_threshold"])
    parser.add_argument("--asr-threshold", type=float, default=run_defaults["coev"]["asr_threshold"])
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
        "--max-metric-calls",
        str(args.max_metric_calls),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
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
        "--results-dir",
        args.coev_results_dir,
        "--eval-method",
        args.eval_method,
        "--refusal-threshold",
        str(args.refusal_threshold),
        "--asr-threshold",
        str(args.asr_threshold),
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
    coev_v2_mode = args.coev_v2_mode
    if coev_v2_mode is None:
        # Backward compatibility: infer coev_v2 mode from legacy coev-mode flag.
        coev_v2_mode = "coev" if args.coev_mode in {"reinforce", "gepa"} else "eval"

    cmd = [
        sys.executable,
        str(COEV_V2_SCRIPT),
        "--mode",
        coev_v2_mode,
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
        "--results-dir",
        args.coev_results_dir,
        "--max-metric-calls",
        str(args.max_metric_calls),
        "--max-new-tokens",
        str(args.max_tokens),
        "--gepa-max-tokens",
        str(args.max_tokens),
        "--gepa-temperature",
        str(args.temperature),
        "--eval-method",
        args.eval_method,
        "--refusal-threshold",
        str(args.refusal_threshold),
        "--asr-threshold",
        str(args.asr_threshold),
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
