"""Shared YAML defaults loader for run scripts."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

# Keys filled from ``global.model_defaults`` when absent under ``runs.<name>``.
_MODEL_DEFAULT_KEYS: tuple[str, ...] = (
    "temperature",
    "baseline_system_prompt",
    "target_system_prompt",
    "seed_prompt",
    "initial_defense_prompt",
    "initial_attacker_instruction",
    "attacker_instruction",
    "eval_instruction",
    "max_new_tokens",
    "max_tokens",
    "gepa_max_tokens",
    "gepa_temperature",
)

_COEV_NESTED_INITIAL_KEYS: tuple[str, ...] = (
    "initial_attacker_instruction",
    "initial_defense_prompt",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"
_CONFIG_ENV_VAR = "PROMPT_OPT_CONFIG_PATH"


def resolve_config_path(path: Path | None = None) -> Path:
    """Path to the active YAML (``PROMPT_OPT_CONFIG_PATH`` or ``configs/default.yaml``)."""
    env_config_path = os.environ.get(_CONFIG_ENV_VAR)
    return path or (Path(env_config_path).expanduser().resolve() if env_config_path else _DEFAULT_CONFIG_PATH)


def load_default_config(path: Path | None = None) -> dict[str, Any]:
    """Load config YAML into a dictionary (raw ``runs.*``; use :func:`merged_run_defaults` for model prompts/temps)."""
    config_path = resolve_config_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must decode to a mapping: {config_path}")
    return payload


def merged_run_defaults(defaults: dict[str, Any], run_key: str) -> dict[str, Any]:
    """Merge ``global.model_defaults`` into ``runs.<run_key>`` (per-run block wins).

    Shared prompts, sampling temperature, and token limits live in ``global.model_defaults``;
    iteration counts, learning rates, paths, and other run-specific knobs stay under ``runs.*``.

    For ``runs.coev``, the ``gepa`` sub-mapping receives ``initial_*`` defaults when omitted there
    (the ``reinforce`` block does not take those keys).

    ``seed_prompt`` defaults to ``target_system_prompt`` when unset.
    ``gepa_temperature`` defaults to ``temperature`` when unset (after merge).
    """
    global_block = defaults.get("global")
    md: dict[str, Any] = {}
    if isinstance(global_block, dict):
        raw_md = global_block.get("model_defaults")
        if isinstance(raw_md, dict):
            md = raw_md
    runs = defaults.get("runs")
    if not isinstance(runs, dict) or run_key not in runs:
        raise KeyError(f"runs.{run_key} not found in config")
    run_raw = runs[run_key]
    if not isinstance(run_raw, dict):
        raise TypeError(f"runs.{run_key} must be a mapping")
    out = copy.deepcopy(run_raw)
    for k in _MODEL_DEFAULT_KEYS:
        if k in md and k not in out:
            out[k] = md[k]
    if run_key == "coev":
        # Only ``gepa`` uses ``initial_*`` (``reinforce`` ReinforceConfig has no such fields).
        sub = "gepa"
        if sub in out and isinstance(out[sub], dict):
            for k in _COEV_NESTED_INITIAL_KEYS:
                if k in md and k not in out[sub]:
                    out[sub][k] = md[k]
    if "seed_prompt" not in out and "target_system_prompt" in out:
        out["seed_prompt"] = out["target_system_prompt"]
    if "gepa_temperature" not in out and "temperature" in out:
        out["gepa_temperature"] = out["temperature"]
    return out


def build_config_snapshot(
    defaults: dict[str, Any],
    *,
    cli_args: Any | None = None,
) -> dict[str, Any]:
    """Lightweight manifest payload: CLI + config file path + env (no embedded YAML).

    Run scripts merge :func:`load_default_config` with argparse; **effective values**
    are captured in ``cli_args`` (after defaults are applied at parse time). The full
    YAML is not duplicated here.

    ``defaults`` is only used for :func:`~src.runtime.target_factory.resolve_reflection_env_overrides`
    so the snapshot records the effective reflection base URL.

    If ``cli_args`` is an :class:`argparse.Namespace`, it is JSON-serialized (with
    ``default=str`` for Path-like values) under ``cli_args``.
    """
    from .target_factory import resolve_reflection_env_overrides

    resolved = resolve_config_path()
    base_url, _api_key = resolve_reflection_env_overrides(defaults)
    out: dict[str, Any] = {
        "effective_config_path": str(resolved.resolve()),
        "repository_default_config_path": str(_DEFAULT_CONFIG_PATH.resolve()),
        "prompt_opt_config_path_env": os.environ.get(_CONFIG_ENV_VAR),
        "environment": {
            "REFLECTION_VLLM_BASE_URL": os.environ.get("REFLECTION_VLLM_BASE_URL"),
            "REFLECTION_VLLM_API_KEY_set": "REFLECTION_VLLM_API_KEY" in os.environ,
            "JUDGE_LOAD_IN_4BIT": os.environ.get("JUDGE_LOAD_IN_4BIT"),
        },
        "effective_reflection_base_url": base_url,
    }
    if cli_args is not None:
        out["cli_args"] = json.loads(json.dumps(vars(cli_args), default=str))
    return out
