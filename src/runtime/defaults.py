"""Shared YAML defaults loader for run scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"
_CONFIG_ENV_VAR = "PROMPT_OPT_CONFIG_PATH"


def resolve_config_path(path: Path | None = None) -> Path:
    """Path to the active YAML (``PROMPT_OPT_CONFIG_PATH`` or ``configs/default.yaml``)."""
    env_config_path = os.environ.get(_CONFIG_ENV_VAR)
    return path or (Path(env_config_path).expanduser().resolve() if env_config_path else _DEFAULT_CONFIG_PATH)


def load_default_config(path: Path | None = None) -> dict[str, Any]:
    """Load repository default config YAML into a dictionary."""
    config_path = resolve_config_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must decode to a mapping: {config_path}")
    return payload


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
