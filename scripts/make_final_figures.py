#!/usr/bin/env python3
"""Generate final-paper figures from saved experiment artifacts.

Outputs are written to ``figures/`` by default:

* adversary_training_trajectory.pdf
* coev_dynamics.pdf
* xstest_comparison.pdf
* safety_utility_pareto.pdf

The script intentionally reads only aggregate logs/metrics. It does not render
raw prompts or model generations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


ADVERSARY_RUNS = [
    (
        "REINFORCE",
        "results/adversary_baseline_test/adversary_training_log.csv",
        "#7a7a7a",
    ),
    (
        "RLOO + LP",
        "results/adversary_rloo_length_penalty/adversary_training_log.csv",
        "#0072b2",
    ),
    (
        "Full seed",
        "results/r11_full_prompt/adversary_training_log.csv",
        "#d55e00",
    ),
    (
        "Full seed, seed 123",
        "results/r11_full_prompt_seed123/adversary_training_log.csv",
        "#009e73",
    ),
    (
        "Full seed + KL",
        "results/r12_full_prompt_kl/adversary_training_log.csv",
        "#cc79a7",
    ),
]


COEV_RUNS = [
    (
        "No KL",
        "results/r14_coev_full_prompt/coev_v2_stage_metrics.csv",
        "results/r14_coev_full_prompt/baseline_vs_optimized_metrics.csv",
        "#0072b2",
    ),
    (
        "KL = 0.05",
        "results/r14_coev_full_prompt_kl/coev_v2_stage_metrics.csv",
        "results/r14_coev_full_prompt_kl/baseline_vs_optimized_metrics.csv",
        "#d55e00",
    ),
]


XSTEST_RUNS = [
    (
        "Default target",
        "target-only",
        "results/xstest_batch_baseline/xstest_summary.json",
        {"safe_compliance": 0.916, "overrefusal_rate": 0.084, "unsafe_asr": 0.075},
    ),
    (
        "GEPA defense",
        "target-only",
        "results/xstest_gepa_defense_target_only/xstest_summary.json",
        {"safe_compliance": 0.740, "overrefusal_rate": 0.260, "unsafe_asr": 0.030},
    ),
    (
        "R11 full seed",
        "adversary",
        "results/r11_seed123_xstest/xstest_summary.json",
        None,
    ),
    (
        "R12 short seed + KL",
        "adversary",
        "results/r12_xstest_eval/xstest_summary.json",
        None,
    ),
    (
        "R12 full seed + KL",
        "adversary",
        "results/r12_full_prompt_kl_xstest/xstest_summary.json",
        None,
    ),
    (
        "R14 coev",
        "adversary",
        "results/r14_coev_full_prompt_xstest/xstest_summary.json",
        None,
    ),
    (
        "R14 coev + KL",
        "adversary",
        "results/r14_coev_full_prompt_kl_xstest/xstest_summary.json",
        None,
    ),
]


SAFETY_UTILITY_POINTS = [
    {
        "label": "Default target prompt",
        "harmbench_asr": 0.42,
        "xstest_overrefusal": 0.084,
        "color": "#0072b2",
    },
    {
        "label": "GEPA defense prompt",
        "harmbench_asr": 0.06,
        "xstest_overrefusal": 0.260,
        "color": "#d55e00",
    },
]


def _artifact(path: str) -> Path:
    return REPO_ROOT / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_adversary_training(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    plotted = 0
    for label, rel_path, color in ADVERSARY_RUNS:
        path = _artifact(rel_path)
        if not path.exists():
            print(f"skip missing adversary log: {rel_path}")
            continue

        rows = _read_csv(path)
        points: list[tuple[int, float]] = []
        for row in rows:
            asr = _maybe_float(row.get("eval_asr"))
            iteration = _maybe_float(row.get("iteration"))
            if asr is None or iteration is None:
                continue
            points.append((int(iteration), asr))

        if not points:
            print(f"skip adversary log with no eval_asr points: {rel_path}")
            continue

        points = sorted(set(points))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.8, label=label, color=color)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No adversary training curves could be plotted.")

    ax.set_title("Adversary Training Trajectories")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Held-out HarmBench ASR")
    ax.set_ylim(-0.02, 0.58)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "adversary_training_trajectory.pdf")
    fig.savefig(out_dir / "adversary_training_trajectory.png", dpi=200)
    plt.close(fig)


def _read_final_optimized_asr(path: Path) -> float | None:
    if not path.exists():
        return None
    for row in _read_csv(path):
        variant = (row.get("variant") or row.get("name") or "").strip().lower()
        if variant == "optimized":
            return _maybe_float(row.get("asr"))
    return None


def plot_coev_dynamics(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    plotted = 0
    for label, stage_rel, final_rel, color in COEV_RUNS:
        stage_path = _artifact(stage_rel)
        if not stage_path.exists():
            print(f"skip missing coev stage metrics: {stage_rel}")
            continue

        points: list[tuple[float, float]] = []
        for row in _read_csv(stage_path):
            if row.get("phase") != "pre_evolution":
                continue
            stage = _maybe_float(row.get("stage"))
            asr = _maybe_float(row.get("asr"))
            if stage is not None and asr is not None:
                points.append((stage, asr))

        final_asr = _read_final_optimized_asr(_artifact(final_rel))
        if final_asr is not None:
            points.append((4.0, final_asr))

        if not points:
            print(f"skip coev run with no ASR points: {stage_rel}")
            continue

        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2.0, label=label, color=color)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No co-evolution curves could be plotted.")

    ax.set_title("Co-evolution Reduces Held-out ASR")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Held-out HarmBench ASR")
    ax.set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "final"])
    ax.set_ylim(-0.02, 0.48)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "coev_dynamics.pdf")
    fig.savefig(out_dir / "coev_dynamics.png", dpi=200)
    plt.close(fig)


def _read_xstest_metrics(rel_path: str, fallback: dict[str, float] | None) -> dict[str, float] | None:
    path = _artifact(rel_path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload)
        return {
            "safe_compliance": float(metrics["safe_compliance"]),
            "overrefusal_rate": float(metrics["overrefusal_rate"]),
            "unsafe_asr": float(metrics["unsafe_asr"]),
        }
    if fallback is not None:
        print(f"use fallback XSTest metrics for missing artifact: {rel_path}")
        return fallback
    print(f"skip missing XSTest summary: {rel_path}")
    return None


def plot_xstest_comparison(out_dir: Path) -> None:
    labels: list[str] = []
    modes: list[str] = []
    safe_compliance: list[float] = []
    overrefusal: list[float] = []
    unsafe_asr: list[float] = []

    for label, mode, rel_path, fallback in XSTEST_RUNS:
        metrics = _read_xstest_metrics(rel_path, fallback)
        if metrics is None:
            continue
        labels.append(label)
        modes.append(mode)
        safe_compliance.append(metrics["safe_compliance"])
        overrefusal.append(metrics["overrefusal_rate"])
        unsafe_asr.append(metrics["unsafe_asr"])

    if not labels:
        raise RuntimeError("No XSTest metrics could be plotted.")

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = list(range(len(labels)))
    width = 0.25

    ax.bar([i - width for i in x], safe_compliance, width, label="Safe compliance", color="#009e73")
    ax.bar(x, overrefusal, width, label="Over-refusal", color="#d55e00")
    ax.bar([i + width for i in x], unsafe_asr, width, label="Unsafe ASR", color="#0072b2")

    for i, mode in enumerate(modes):
        if mode == "target-only":
            ax.text(i, 1.04, "target", ha="center", va="bottom", fontsize=7, color="#555555")
        else:
            ax.text(i, 1.04, "adv", ha="center", va="bottom", fontsize=7, color="#555555")

    ax.set_title("XSTest Safety and Utility Metrics")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.23))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "xstest_comparison.pdf")
    fig.savefig(out_dir / "xstest_comparison.png", dpi=200)
    plt.close(fig)


def plot_safety_utility_pareto(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    for point in SAFETY_UTILITY_POINTS:
        ax.scatter(
            point["xstest_overrefusal"],
            point["harmbench_asr"],
            s=85,
            color=point["color"],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            point["label"],
            (point["xstest_overrefusal"], point["harmbench_asr"]),
            xytext=(8, 5),
            textcoords="offset points",
            fontsize=8,
        )

    start = SAFETY_UTILITY_POINTS[0]
    end = SAFETY_UTILITY_POINTS[1]
    ax.annotate(
        "",
        xy=(end["xstest_overrefusal"], end["harmbench_asr"]),
        xytext=(start["xstest_overrefusal"], start["harmbench_asr"]),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2, "alpha": 0.8},
    )

    ax.set_title("Safety-Utility Tradeoff")
    ax.set_xlabel("XSTest over-refusal rate")
    ax.set_ylabel("Held-out HarmBench ASR")
    ax.set_xlim(0.0, 0.32)
    ax.set_ylim(0.0, 0.48)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "safety_utility_pareto.pdf")
    fig.savefig(out_dir / "safety_utility_pareto.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final-paper figures.")
    parser.add_argument("--out-dir", default="figures", help="Output directory for generated figures.")
    parser.add_argument(
        "--only",
        choices=["all", "adversary", "coev", "xstest", "pareto"],
        default="all",
        help="Generate only one figure group.",
    )
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only in {"all", "adversary"}:
        plot_adversary_training(out_dir)
    if args.only in {"all", "coev"}:
        plot_coev_dynamics(out_dir)
    if args.only in {"all", "xstest"}:
        plot_xstest_comparison(out_dir)
    if args.only in {"all", "pareto"}:
        plot_safety_utility_pareto(out_dir)

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
