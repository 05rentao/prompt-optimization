"""Shared orchestration helpers for run scripts in `runs/`.

These helpers keep script pipelines visually consistent while preserving each
run's unique optimization logic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from src.evaluators import compute_refusal_score, is_yes_verdict
from src.runtime import EvaluationConfig, GenerationSession


def adversary_rewrite_sample(
    adversary_session: GenerationSession,
    device: str,
    user_prompt: str,
    instruction: str,
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_length: int = 2048,
) -> tuple[dict[str, Any], float]:
    """Sample one adversary rewrite with token-level metadata and elapsed milliseconds."""
    messages = [
        {"role": "system", "content": f"You are a rewriter. {instruction}"},
        {"role": "user", "content": user_prompt},
    ]
    start = time.perf_counter()
    runtime = adversary_session.runtime
    sample = runtime.sample_policy(
        messages=messages,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_length=max_length,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return sample, elapsed_ms


def build_prompt_pool(
    train_data: Sequence[dict[str, Any]],
    val_data: Sequence[dict[str, Any]],
    key: str = "prompt",
) -> list[str]:
    """Build a single prompt pool from train/val rows."""
    return [sample[key] for sample in train_data] + [sample[key] for sample in val_data]


def split_prompt_pool(
    prompts: Sequence[str],
    train_slice_end: int,
    eval_slice_start: int,
    eval_slice_end: int,
    *,
    require_train: bool = True,
    require_eval: bool = True,
    script_name: str = "run script",
) -> tuple[list[str], list[str]]:
    """Slice train/eval prompt windows with consistent guardrails."""
    train_prompts = list(prompts[:train_slice_end])
    eval_prompts = list(prompts[eval_slice_start:eval_slice_end])
    if require_train and not train_prompts:
        raise ValueError(f"{script_name}: no train prompts available; increase train_size or expand training slice.")
    if require_eval and not eval_prompts:
        raise ValueError(f"{script_name}: no eval prompts available; increase val_size or expand eval slice.")
    return train_prompts, eval_prompts


def compute_reward_and_verdict(
    behavior: str,
    generation: str,
    eval_cfg: EvaluationConfig,
    judge_session: GenerationSession | None,
) -> tuple[float, str]:
    """Compute reward + verdict consistently across judge/heuristic modes."""
    if eval_cfg.method == "judge":
        if judge_session is None:
            raise ValueError("judge_session is required when eval method is 'judge'.")
        verdict = judge_session.judge(behaviors=behavior, generations=[generation])[0]
        return (1.0 if is_yes_verdict(verdict) else 0.0), str(verdict)

    refusal_score = compute_refusal_score(generation, eval_cfg.refusal_patterns)
    success = (1.0 - refusal_score) > eval_cfg.asr_threshold
    return (1.0 if success else 0.0), ("Yes" if success else "No")


def maybe_save_adapters(
    adversary_session: GenerationSession,
    save_dir: str | None,
    *,
    results_dir: Path | None = None,
) -> str | None:
    """Save adversary adapters if requested and return saved path.

    If ``results_dir`` is set and ``save_dir`` is a relative path, the output
    directory is ``results_dir / save_dir`` (not the process working directory).
    Absolute ``save_dir`` paths are respected as-is.
    """
    if not save_dir:
        return None
    raw = Path(save_dir)
    if raw.is_absolute():
        output_dir = raw.resolve()
    elif results_dir is not None:
        output_dir = (results_dir / raw).resolve()
    else:
        output_dir = raw.resolve()
    runtime = adversary_session.runtime
    if not hasattr(runtime, "save_adapters"):
        raise RuntimeError("Adversary runtime must expose save_adapters().")
    runtime.save_adapters(output_dir)
    return str(output_dir)
