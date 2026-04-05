from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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


def format_duration_human(seconds: float) -> str:
    """Compact wall-clock label for metrics JSON, e.g. ``45s``, ``12m 30s``, ``1h 5m 2s``."""

    total = int(round(max(0.0, float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return " ".join(parts)


def record_run_timing(
    repo_root: Path,
    results_dir: Path,
    *,
    script: str,
    run_start: float,
    run_seconds: float,
    extra: dict[str, Any] | None = None,
    include_argv: bool = True,
) -> tuple[Path, Path]:
    """Persist wall-clock timing for a run: per-directory JSON + append-only repo log.

    Writes ``results_dir/run_timing.json`` and appends one JSON object per line to
    ``<repo>/results/experiment_timing_log.jsonl`` so you can grep historical run
    durations without opening each result folder.
    """
    ended = datetime.now(timezone.utc)
    started = datetime.fromtimestamp(run_start, tz=timezone.utc)
    payload: dict[str, Any] = {
        "script": script,
        "wall_seconds": round(float(run_seconds), 3),
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "results_dir": str(results_dir.resolve()),
    }
    if extra:
        payload["extra"] = extra
    if include_argv:
        payload["argv"] = sys.argv[:128]
    timing_path = write_json(results_dir / "run_timing.json", payload)
    log_path = repo_root / "results" / "experiment_timing_log.jsonl"
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
    return timing_path, log_path


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
    *,
    baseline_variant: str = "baseline",
    comparison_variant: str = "optimized",
) -> pd.DataFrame:
    """Construct canonical baseline-vs-comparison metrics dataframe.

    Defaults match GEPA/CoEV (defense prompt tuning). For adversary-only runs,
    pass explicit ``baseline_variant`` / ``comparison_variant`` (e.g. before/after
    weight training) so ``optimized`` is not misread as prompt optimization.
    """
    return pd.DataFrame(
        [
            {"variant": baseline_variant, **baseline_metrics},
            {"variant": comparison_variant, **optimized_metrics},
        ]
    )


def save_baseline_optimized_plot(
    comparison_df: pd.DataFrame,
    out_path: Path,
    title: str,
    key_metrics: tuple[str, ...] = ("refusal_rate", "asr", "aggregate_score"),
    *,
    subtitle: str | None = None,
) -> Path:
    """Render and save baseline-vs-comparison bar chart."""
    plot_df = comparison_df.melt(
        id_vars=["variant"],
        value_vars=list(key_metrics),
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metric", y="value", hue="variant")
    plt.ylim(0, 1)
    if subtitle:
        plt.title(f"{title}\n{subtitle}", fontsize=10)
    else:
        plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def save_gepa_refusal_vs_evaluator_calls_plot(
    trace_df: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    score_col: str = "score",
) -> Path | None:
    """Plot per-call refusal and cumulative best refusal from the GEPA trace (defense objective).

    ``score`` is the evaluator return: 1.0 = refusal/safe, 0.0 = harmful completion
    (judge or heuristic). This matches ``plot_optimization_trajectory`` semantics;
    the black line should trend upward when optimization finds safer behavior.
    """
    if trace_df.empty or score_col not in trace_df.columns:
        return None
    frame = trace_df.reset_index(drop=True).copy()
    frame["call_index"] = frame.index + 1
    frame["refusal"] = pd.to_numeric(frame[score_col], errors="coerce").clip(0.0, 1.0)
    frame["best_refusal_so_far"] = frame["refusal"].cummax()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=frame, x="call_index", y="refusal", label="per-call refusal", alpha=0.65)
    sns.lineplot(
        data=frame,
        x="call_index",
        y="best_refusal_so_far",
        label="best refusal so far",
        color="black",
    )
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Evaluator call")
    plt.ylabel("Refusal rate")
    plt.legend(loc="best")
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


def save_coev_refusal_vs_global_step_plot(
    stage_metrics_df: pd.DataFrame,
    iters_per_stage: int,
    out_path: Path,
    title: str,
) -> Path | None:
    """Plot refusal rate at CoEV checkpoints vs global step (defense-aligned complement to ASR plot)."""
    if stage_metrics_df.empty or "stage" not in stage_metrics_df.columns:
        return None
    if "phase" not in stage_metrics_df.columns or "refusal_rate" not in stage_metrics_df.columns:
        return None

    phases = ("pre_evolution", "attacker_gepa_best", "defender_gepa_best")
    sub = stage_metrics_df[stage_metrics_df["phase"].isin(phases)].copy()
    sub["refusal_rate"] = pd.to_numeric(sub["refusal_rate"], errors="coerce")
    sub = sub.dropna(subset=["refusal_rate"])
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
    sns.lineplot(data=sub, x="global_step", y="refusal_rate", hue="phase", marker="o")
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Global training step (end of stage bundle + offset for GEPA evals)")
    plt.ylabel("Refusal rate")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def save_adversary_refusal_vs_iterations_plot(
    training_df: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    final_refusal: float | None = None,
    iterations: int | None = None,
) -> Path | None:
    """Plot periodic eval refusal rate vs training iteration (1 - ASR; target defense behavior)."""
    if training_df.empty:
        return None
    if "eval_asr" not in training_df.columns or "iteration" not in training_df.columns:
        return None
    frame = training_df.copy()
    frame["eval_asr_num"] = pd.to_numeric(frame["eval_asr"], errors="coerce")
    pts = frame.dropna(subset=["eval_asr_num"])
    if pts.empty and final_refusal is None:
        return None

    plt.figure(figsize=(10, 5))
    if not pts.empty:
        pts = pts.copy()
        pts["eval_refusal_num"] = 1.0 - pts["eval_asr_num"]
        sns.lineplot(data=pts, x="iteration", y="eval_refusal_num", marker="o", label="periodic eval")
    if final_refusal is not None and iterations is not None and iterations > 0:
        plt.scatter(
            [iterations],
            [final_refusal],
            color="tab:red",
            s=80,
            zorder=5,
            label="final eval",
        )
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Training iteration")
    plt.ylabel("Refusal rate")
    plt.legend(loc="best")
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
    """Render and save optimization trajectory plot if trace is non-empty.

    Single-role (no ``hue_col``): ``score`` is the objective (e.g. refusal); black line is
    cumulative best so far.

    Dual-role (e.g. CoEV attacker vs defender): ``score`` means attack success for
    ``role=attacker`` and refusal for ``role=defender``. X-axis is **within-role**
    evaluator index; cumulative best is computed **per role** (not across a single
    mixed global index).
    """
    if trace_df.empty:
        return None
    frame = trace_df.reset_index(drop=True).copy()
    if hue_col and hue_col in frame.columns:
        frame["role_call_index"] = frame.groupby(hue_col, sort=False).cumcount() + 1
        frame["best_so_far"] = frame.groupby(hue_col, sort=False)[score_col].transform(lambda s: s.cummax())
        long = pd.concat(
            [
                frame.assign(_series="per-call", _y=frame[score_col]),
                frame.assign(_series="best so far", _y=frame["best_so_far"]),
            ],
            ignore_index=True,
        )
        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=long,
            x="role_call_index",
            y="_y",
            hue=hue_col,
            style="_series",
            alpha=0.85,
        )
        plt.xlabel("Evaluator call (within role)")
        plt.ylabel("Objective (per role)")
    else:
        frame["call_index"] = frame.index + 1
        frame["best_so_far"] = frame[score_col].cummax()
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=frame, x="call_index", y=score_col, label="raw score")
        sns.lineplot(data=frame, x="call_index", y="best_so_far", label="best so far", color="black")
        plt.xlabel("Evaluator call")
        plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend(loc="best")
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return out_path
