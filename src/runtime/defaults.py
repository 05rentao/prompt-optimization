"""Shared YAML defaults loader for run scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"
_CONFIG_ENV_VAR = "PROMPT_OPT_CONFIG_PATH"


def load_default_config(path: Path | None = None) -> dict[str, Any]:
    """Load repository default config YAML into a dictionary."""
    env_config_path = os.environ.get(_CONFIG_ENV_VAR)
    config_path = path or (Path(env_config_path).expanduser().resolve() if env_config_path else _DEFAULT_CONFIG_PATH)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must decode to a mapping: {config_path}")
    return payload
