"""Shared YAML defaults loader for run scripts."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"
_CONFIG_ENV_VAR = "PROMPT_OPT_CONFIG_PATH"


def resolve_hf_token() -> str:
    """Resolve HF token from standard environment variable names."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    return token


@contextmanager
def scoped_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Temporarily apply environment variable overrides."""
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def resolve_config_path(path: Path | None = None) -> Path:
    """Path to the active YAML (``PROMPT_OPT_CONFIG_PATH`` or ``configs/default.yaml``)."""
    env_config_path = os.environ.get(_CONFIG_ENV_VAR)
    return path or (Path(env_config_path).expanduser().resolve() if env_config_path else _DEFAULT_CONFIG_PATH)


def deep_merge_runs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings; ``override`` wins on key conflicts."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_runs(out[k], v)
        else:
            out[k] = v
    return out


def apply_shared_generation_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge ``shared_generation`` into each ``runs.<name>`` block (YAML overrides shared).

    Resolution order: ``shared_generation`` defaults, then per-run keys from the file,
    then CLI (handled by run scripts). Missing ``shared_generation`` leaves ``payload`` unchanged.
    """
    sg = payload.get("shared_generation")
    if not sg or not isinstance(sg, dict):
        return payload

    seed = sg.get("seed_attacker_instruction")
    target = sg.get("target_system_prompt")
    samp = sg.get("sampling") if isinstance(sg.get("sampling"), dict) else {}
    eth = sg.get("eval_thresholds") if isinstance(sg.get("eval_thresholds"), dict) else {}

    runs = payload.get("runs")
    if not isinstance(runs, dict):
        return payload

    gepa_base: dict[str, Any] = {}
    if target is not None:
        gepa_base["baseline_system_prompt"] = target
    if "gepa_max_tokens" in samp and samp["gepa_max_tokens"] is not None:
        gepa_base["max_tokens"] = samp["gepa_max_tokens"]
    if "gepa_temperature" in samp and samp["gepa_temperature"] is not None:
        gepa_base["temperature"] = samp["gepa_temperature"]
    if "refusal_threshold" in eth and eth["refusal_threshold"] is not None:
        gepa_base["refusal_threshold"] = eth["refusal_threshold"]
    if "asr_threshold" in eth and eth["asr_threshold"] is not None:
        gepa_base["asr_threshold"] = eth["asr_threshold"]

    coev_base: dict[str, Any] = {}
    if seed is not None:
        coev_base["eval_instruction"] = seed
    for k in ("refusal_threshold", "asr_threshold"):
        if k in eth and eth[k] is not None:
            coev_base[k] = eth[k]
    nested_gepa: dict[str, Any] = {}
    nested_reinforce: dict[str, Any] = {}
    if seed is not None:
        nested_gepa["initial_attacker_instruction"] = seed
        nested_reinforce["attacker_instruction"] = seed
    if target is not None:
        nested_gepa["initial_defense_prompt"] = target
    if nested_gepa:
        coev_base["gepa"] = nested_gepa
    if nested_reinforce:
        coev_base["reinforce"] = nested_reinforce

    coev_v2_base: dict[str, Any] = {}
    if seed is not None:
        coev_v2_base["initial_attacker_instruction"] = seed
    if target is not None:
        coev_v2_base["initial_defense_prompt"] = target
    for yaml_key, samp_key in (
        ("max_new_tokens", "adversary_max_new_tokens"),
        ("gepa_max_tokens", "gepa_max_tokens"),
        ("gepa_temperature", "gepa_temperature"),
    ):
        if samp_key in samp and samp[samp_key] is not None:
            coev_v2_base[yaml_key] = samp[samp_key]
    for k in ("refusal_threshold", "asr_threshold"):
        if k in eth and eth[k] is not None:
            coev_v2_base[k] = eth[k]

    adversary_base: dict[str, Any] = {}
    if seed is not None:
        adversary_base["attacker_instruction"] = seed
    if target is not None:
        adversary_base["target_system_prompt"] = target
    for yaml_key, samp_key in (
        ("max_new_tokens", "adversary_max_new_tokens"),
        ("target_max_new_tokens", "target_max_new_tokens"),
    ):
        if samp_key in samp and samp[samp_key] is not None:
            adversary_base[yaml_key] = samp[samp_key]
    for k in ("refusal_threshold", "asr_threshold"):
        if k in eth and eth[k] is not None:
            adversary_base[k] = eth[k]

    vsb_base: dict[str, Any] = {}
    if target is not None:
        vsb_base["seed_prompt"] = target
    if "adversary_max_new_tokens" in samp and samp["adversary_max_new_tokens"] is not None:
        vsb_base["max_new_tokens"] = samp["adversary_max_new_tokens"]
    if "target_temperature" in samp and samp["target_temperature"] is not None:
        vsb_base["temperature"] = samp["target_temperature"]

    merged_runs = deepcopy(runs)
    for name, base in (
        ("gepa", gepa_base),
        ("coev", coev_base),
        ("coev_v2", coev_v2_base),
        ("adversary", adversary_base),
        ("vector_steering_baseline", vsb_base),
    ):
        if not base:
            continue
        if name in merged_runs and isinstance(merged_runs[name], dict):
            merged_runs[name] = deep_merge_runs(base, merged_runs[name])

    out = dict(payload)
    out["runs"] = merged_runs
    return out


def load_default_config(path: Path | None = None) -> dict[str, Any]:
    """Load repository default config YAML into a dictionary.

    Applies :func:`apply_shared_generation_defaults` so ``shared_generation`` seeds
    per-run blocks; values in YAML still override shared defaults.
    """
    config_path = resolve_config_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must decode to a mapping: {config_path}")
    return apply_shared_generation_defaults(payload)


def build_config_snapshot(
    defaults: dict[str, Any],
    *,
    cli_args: Any | None = None,
) -> dict[str, Any]:
    """Lightweight manifest payload: CLI + config file path + env (no embedded YAML).

    Run scripts merge :func:`load_default_config` with argparse; **effective values**
    are captured in ``cli_args`` (after defaults are applied at parse time). The full
    YAML is not duplicated here.

    ``defaults`` is only used for :func:`~src.runtime.sessions.resolve_reflection_env_overrides`
    so the snapshot records the effective reflection base URL.

    If ``cli_args`` is an :class:`argparse.Namespace`, it is JSON-serialized (with
    ``default=str`` for Path-like values) under ``cli_args``.
    """
    from .sessions import resolve_reflection_env_overrides

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
