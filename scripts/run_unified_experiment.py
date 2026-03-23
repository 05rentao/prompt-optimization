#!/usr/bin/env python3
"""Unified runner for CoEV and GEPA experiments.

Dispatches to the existing ``runs/*.py`` entrypoints. The only CLI flag is
``--mode``; everything else comes from the active config YAML
(``configs/default.yaml`` or ``PROMPT_OPT_CONFIG_PATH``), under
``scripts.unified_runner`` and ``global`` / ``runs.*``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.runtime.defaults import load_default_config


REPO_ROOT = Path(__file__).resolve().parent.parent
GEPA_SCRIPT = REPO_ROOT / "runs" / "gepa_run.py"
COEV_SCRIPT = REPO_ROOT / "runs" / "coev_run.py"
COEV_V2_SCRIPT = REPO_ROOT / "runs" / "coev_v2_run.py"
ADVERSARY_SCRIPT = REPO_ROOT / "runs" / "adversary_run.py"


def _resolve_orchestration(defaults: dict[str, Any]) -> dict[str, Any]:
    """Read scripts.unified_runner and apply run_kind / mode defaults."""

    u = defaults["scripts"]["unified_runner"]
    run_kind = u.get("run_kind", "train")
    coev_mode = u.get("coev_mode", "reinforce")
    coev_v2_mode = u.get("coev_v2_mode")
    adversary_mode = u.get("adversary_mode", "train")

    if run_kind == "eval":
        coev_mode = "eval"
        coev_v2_mode = "eval"
        adversary_mode = "eval"

    if coev_v2_mode is None:
        coev_v2_mode = "coev" if coev_mode in {"reinforce", "gepa"} else "eval"

    gepa_results_dir = u.get("gepa_results_dir", "results/gepa")
    coev_results_dir = u["coev_results_dir"]
    coev_v2_results_dir = u.get("coev_v2_results_dir", coev_results_dir)
    adversary_results_dir = u.get("adversary_results_dir", "results/adversary")

    save_dir = u.get("save_dir")
    gepa_show_progress = u.get("gepa_show_progress", True)

    device = defaults["global"].get("device")

    return {
        "coev_mode": coev_mode,
        "coev_v2_mode": coev_v2_mode,
        "adversary_mode": adversary_mode,
        "gepa_results_dir": gepa_results_dir,
        "coev_results_dir": coev_results_dir,
        "coev_v2_results_dir": coev_v2_results_dir,
        "adversary_results_dir": adversary_results_dir,
        "save_dir": save_dir,
        "gepa_show_progress": gepa_show_progress,
        "device": device,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CoEV/GEPA experiment runner.")
    parser.add_argument("--mode", choices=["gepa", "coev", "coev_v2", "adversary"], required=True)
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build_gepa_command(o: dict[str, Any]) -> list[str]:
    cmd: list[str] = [sys.executable, str(GEPA_SCRIPT), "--results-dir", o["gepa_results_dir"]]
    if o["gepa_show_progress"]:
        cmd.append("--show-progress")
    if o["device"]:
        cmd.extend(["--device", str(o["device"])])
    return cmd


def build_coev_command(o: dict[str, Any]) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(COEV_SCRIPT),
        "--mode",
        o["coev_mode"],
        "--results-dir",
        o["coev_results_dir"],
    ]
    if o["device"]:
        cmd.extend(["--device", str(o["device"])])
    if o.get("save_dir"):
        cmd.extend(["--save-dir", str(o["save_dir"])])
    return cmd


def build_coev_v2_command(o: dict[str, Any]) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(COEV_V2_SCRIPT),
        "--mode",
        o["coev_v2_mode"],
        "--results-dir",
        o["coev_v2_results_dir"],
    ]
    if o["device"]:
        cmd.extend(["--device", str(o["device"])])
    if o.get("save_dir"):
        cmd.extend(["--save-dir", str(o["save_dir"])])
    return cmd


def build_adversary_command(o: dict[str, Any]) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(ADVERSARY_SCRIPT),
        "--mode",
        o["adversary_mode"],
        "--results-dir",
        o["adversary_results_dir"],
    ]
    if o["device"]:
        cmd.extend(["--device", str(o["device"])])
    if o.get("save_dir"):
        cmd.extend(["--save-dir", str(o["save_dir"])])
    return cmd


def main() -> None:
    args = parse_args()
    defaults = load_default_config()
    o = _resolve_orchestration(defaults)

    if args.mode == "gepa":
        run_command(build_gepa_command(o))
        return

    if args.mode == "coev":
        run_command(build_coev_command(o))
        return

    if args.mode == "coev_v2":
        run_command(build_coev_v2_command(o))
        return

    if args.mode == "adversary":
        run_command(build_adversary_command(o))


if __name__ == "__main__":
    main()
