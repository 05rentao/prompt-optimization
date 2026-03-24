from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .types import RunManifest


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON payload to disk with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_text(path: Path, content: str) -> Path:
    """Write UTF-8 text content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    """Write dataframe to CSV without index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_many_csv(base_dir: Path, frames: dict[str, pd.DataFrame], skip_empty: set[str] | None = None) -> dict[str, Path]:
    """Write multiple dataframe artifacts under one directory."""
    ensure_dir(base_dir)
    skip_empty = skip_empty or set()
    outputs: dict[str, Path] = {}
    for filename, frame in frames.items():
        if filename in skip_empty and frame.empty:
            continue
        out_path = write_csv(base_dir / filename, frame)
        outputs[filename] = out_path
    return outputs


def build_baseline_optimized_df(
    baseline_metrics: dict[str, float],
    optimized_metrics: dict[str, float],
) -> pd.DataFrame:
    """Construct canonical baseline-vs-optimized metrics dataframe."""
    return pd.DataFrame(
        [
            {"variant": "baseline", **baseline_metrics},
            {"variant": "optimized", **optimized_metrics},
        ]
    )


def save_baseline_optimized_plot(
    comparison_df: pd.DataFrame,
    out_path: Path,
    title: str,
    key_metrics: tuple[str, ...] = ("refusal_rate", "asr", "aggregate_score"),
) -> Path:
    """Render and save baseline-vs-optimized bar chart."""
    plot_df = comparison_df.melt(
        id_vars=["variant"],
        value_vars=list(key_metrics),
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="variant")
    plt.ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def save_gepa_asr_vs_evaluator_calls_plot(
    trace_df: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    score_col: str = "score",
) -> Path | None:
    """Plot ASR vs evaluator call index from GEPA trace (standalone: score is refusal rate)."""
    if trace_df.empty or score_col not in trace_df.columns:
        return None
    frame = trace_df.reset_index(drop=True).copy()
    frame["call_index"] = frame.index + 1
    frame["asr"] = (1.0 - pd.to_numeric(frame[score_col], errors="coerce")).clip(0.0, 1.0)
    frame["best_asr_so_far"] = frame["asr"].cummax()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=frame, x="call_index", y="asr", label="per-call ASR", alpha=0.65)
    sns.lineplot(data=frame, x="call_index", y="best_asr_so_far", label="best ASR so far", color="black")
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Evaluator call")
    plt.ylabel("ASR")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def save_adversary_asr_vs_iterations_plot(
    training_df: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    final_asr: float | None = None,
    iterations: int | None = None,
) -> Path | None:
    """Plot periodic eval ASR vs training iteration from adversary_training_log (sparse eval_asr)."""
    if training_df.empty:
        return None
    if "eval_asr" not in training_df.columns or "iteration" not in training_df.columns:
        return None
    frame = training_df.copy()
    frame["eval_asr_num"] = pd.to_numeric(frame["eval_asr"], errors="coerce")
    pts = frame.dropna(subset=["eval_asr_num"])
    if pts.empty and final_asr is None:
        return None

    plt.figure(figsize=(10, 5))
    if not pts.empty:
        sns.lineplot(data=pts, x="iteration", y="eval_asr_num", marker="o", label="periodic eval")
    if final_asr is not None and iterations is not None and iterations > 0:
        plt.scatter(
            [iterations],
            [final_asr],
            color="tab:red",
            s=80,
            zorder=5,
            label="final eval",
        )
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Training iteration")
    plt.ylabel("ASR")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def save_coev_asr_vs_global_step_plot(
    stage_metrics_df: pd.DataFrame,
    iters_per_stage: int,
    out_path: Path,
    title: str,
) -> Path | None:
    """Plot ASR at CoEV checkpoints vs global REINFORCE step (end-of-stage + post-GEPA evals)."""
    if stage_metrics_df.empty or "stage" not in stage_metrics_df.columns:
        return None
    if "phase" not in stage_metrics_df.columns or "asr" not in stage_metrics_df.columns:
        return None

    phases = ("pre_evolution", "attacker_gepa_best", "defender_gepa_best")
    sub = stage_metrics_df[stage_metrics_df["phase"].isin(phases)].copy()
    sub["asr"] = pd.to_numeric(sub["asr"], errors="coerce")
    sub = sub.dropna(subset=["asr"])
    if sub.empty:
        return None

    ips = max(1, int(iters_per_stage))

    def _x_for_row(row: pd.Series) -> float:
        st = int(row["stage"])
        ph = str(row["phase"])
        base = (st + 1) * ips
        if ph == "pre_evolution":
            return float(base)
        if ph == "attacker_gepa_best":
            return float(base) + 0.33
        if ph == "defender_gepa_best":
            return float(base) + 0.66
        return float(base)

    sub["global_step"] = sub.apply(_x_for_row, axis=1)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=sub, x="global_step", y="asr", hue="phase", marker="o")
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Global training step (end of stage bundle + offset for GEPA evals)")
    plt.ylabel("ASR")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def save_trajectory_plot(
    trace_df: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    score_col: str = "score",
    hue_col: str | None = None,
) -> Path | None:
    """Render and save optimization trajectory plot if trace is non-empty."""
    if trace_df.empty:
        return None
    frame = trace_df.reset_index(drop=True).copy()
    frame["call_index"] = frame.index + 1
    frame["best_so_far"] = frame[score_col].cummax()
    plt.figure(figsize=(10, 5))
    if hue_col:
        sns.lineplot(data=frame, x="call_index", y=score_col, hue=hue_col, alpha=0.65)
    else:
        sns.lineplot(data=frame, x="call_index", y=score_col, label="raw score")
    sns.lineplot(data=frame, x="call_index", y="best_so_far", label="best so far", color="black")
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Evaluator call")
    plt.ylabel("Score")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def log_saved_artifacts(paths: Iterable[Path]) -> None:
    """Print normalized artifact locations to stdout."""
    print("Saved artifacts:")
    for path in paths:
        print(" -", path)


def write_run_manifest(results_dir: Path, payload: RunManifest | dict[str, Any]) -> Path:
    ensure_dir(results_dir)
    out_path = results_dir / "run_manifest.json"

    if isinstance(payload, RunManifest):
        body = payload.to_dict()
    else:
        body = payload

    return write_json(out_path, body)
