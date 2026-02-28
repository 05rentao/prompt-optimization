"""Load experiment config from YAML. Single source of truth for config path."""
import yaml
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = "configs/default.yaml"


def load_config(path: str = None) -> Dict[str, Any]:
    """
    Load experiment config from a YAML file.

    Args:
        path: Path to YAML file. If None, uses configs/default.yaml.

    Returns:
        Config dict (adversary, evaluator, defense, target, training, etc.).

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = path or DEFAULT_CONFIG_PATH
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
