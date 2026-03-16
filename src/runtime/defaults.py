"""Shared YAML defaults loader for run scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"


def load_default_config(path: Path | None = None) -> dict[str, Any]:
    """Load repository default config YAML into a dictionary."""
    config_path = path or _DEFAULT_CONFIG_PATH
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must decode to a mapping: {config_path}")
    return payload
