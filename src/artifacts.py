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
