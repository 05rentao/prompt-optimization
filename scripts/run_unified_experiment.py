#!/usr/bin/env python3
"""Unified runner for GEPA, CoEV v2, CoEV v2 RLOO, and adversary experiments.

Dispatches to ``runs/gepa_run.py``, ``runs/coev_v2_run.py`` (REINFORCE or RLOO via
``--adversary-policy``), and ``runs/adversary_run.py``. The only CLI flag is
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runtime.defaults import load_default_config
GEPA_SCRIPT = REPO_ROOT / "runs" / "gepa_run.py"
COEV_V2_SCRIPT = REPO_ROOT / "runs" / "coev_v2_run.py"
ADVERSARY_SCRIPT = REPO_ROOT / "runs" / "adversary_run.py"


def _resolve_orchestration(defaults: dict[str, Any]) -> dict[str, Any]:
    """Read scripts.unified_runner and apply run_kind / mode defaults."""

    u = defaults["scripts"]["unified_runner"]
    run_kind = u.get("run_kind", "train")
    coev_v2_mode = u.get("coev_v2_mode")
    coev_v2_rloo_mode = u.get("coev_v2_rloo_mode")
    adversary_mode = u.get("adversary_mode", "train")

    if run_kind == "eval":
        coev_v2_mode = "eval"
        coev_v2_rloo_mode = "eval"
        adversary_mode = "eval"

    if coev_v2_mode is None:
        coev_v2_mode = "coev"

    if coev_v2_rloo_mode is None:
        coev_v2_rloo_mode = "coev"

    gepa_results_dir = u.get("gepa_results_dir", "results/gepa")
    coev_v2_results_dir = u.get("coev_v2_results_dir", "results/coev_v2")
    coev_v2_rloo_results_dir = u.get("coev_v2_rloo_results_dir", "results/coev_v2_rloo")
    adversary_results_dir = u.get("adversary_results_dir", "results/adversary")

    save_dir = u.get("save_dir")

    device = defaults["global"].get("device")

    return {
        "coev_v2_mode": coev_v2_mode,
        "coev_v2_rloo_mode": coev_v2_rloo_mode,
        "adversary_mode": adversary_mode,
        "gepa_results_dir": gepa_results_dir,
        "coev_v2_results_dir": coev_v2_results_dir,
        "coev_v2_rloo_results_dir": coev_v2_rloo_results_dir,
        "adversary_results_dir": adversary_results_dir,
        "save_dir": save_dir,
        "device": device,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CoEV/GEPA experiment runner.")
    parser.add_argument(
        "--mode",
        choices=["gepa", "coev_v2", "coev_v2_rloo", "adversary"],
        required=True,
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build_gepa_command(o: dict[str, Any]) -> list[str]:
    cmd: list[str] = [sys.executable, str(GEPA_SCRIPT), "--results-dir", o["gepa_results_dir"]]
    if o["device"]:
        cmd.extend(["--device", str(o["device"])])
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


def build_coev_v2_rloo_command(o: dict[str, Any]) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(COEV_V2_SCRIPT),
        "--mode",
        o["coev_v2_rloo_mode"],
        "--adversary-policy",
        "rloo",
        "--results-dir",
        o["coev_v2_rloo_results_dir"],
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

    if args.mode == "coev_v2":
        run_command(build_coev_v2_command(o))
        return

    if args.mode == "coev_v2_rloo":
        run_command(build_coev_v2_rloo_command(o))
        return

    if args.mode == "adversary":
        run_command(build_adversary_command(o))


if __name__ == "__main__":
    main()
