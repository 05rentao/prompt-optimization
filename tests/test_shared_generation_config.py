"""Tests for shared_generation merge in src/runtime/defaults.py."""

from __future__ import annotations

from pathlib import Path

from src.runtime.defaults import apply_shared_generation_defaults, deep_merge_runs, load_default_config


def test_deep_merge_runs_nested() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 9, "z": 3}, "b": 2}
    assert deep_merge_runs(base, override) == {"a": 1, "b": 2, "nested": {"x": 1, "y": 9, "z": 3}}


def test_apply_shared_generation_merges_into_runs() -> None:
    payload = {
        "shared_generation": {
            "seed_attacker_instruction": "SEED_ATTACKER",
            "target_system_prompt": "TARGET_SYS",
            "sampling": {
                "adversary_max_new_tokens": 111,
                "target_max_new_tokens": 222,
                "target_temperature": 0.0,
                "gepa_max_tokens": 333,
                "gepa_temperature": 0.5,
            },
            "eval_thresholds": {"refusal_threshold": 0.7, "asr_threshold": 0.3},
        },
        "runs": {
            "gepa": {"train_size": 5, "eval_method": "judge"},
            "coev": {
                "reinforce": {"iterations": 2},
                "gepa": {"stages": 1},
            },
            "coev_v2": {"stages": 2},
            "adversary": {},
            "vector_steering_baseline": {"target_layers": [15]},
        },
    }
    out = apply_shared_generation_defaults(payload)
    g = out["runs"]["gepa"]
    assert g["baseline_system_prompt"] == "TARGET_SYS"
    assert g["max_tokens"] == 333
    assert g["temperature"] == 0.5
    assert g["train_size"] == 5
    assert g["eval_method"] == "judge"

    c = out["runs"]["coev"]
    assert c["eval_instruction"] == "SEED_ATTACKER"
    assert c["gepa"]["initial_attacker_instruction"] == "SEED_ATTACKER"
    assert c["gepa"]["initial_defense_prompt"] == "TARGET_SYS"
    assert c["gepa"]["stages"] == 1
    assert c["reinforce"]["attacker_instruction"] == "SEED_ATTACKER"
    assert c["reinforce"]["iterations"] == 2

    v2 = out["runs"]["coev_v2"]
    assert v2["initial_attacker_instruction"] == "SEED_ATTACKER"
    assert v2["initial_defense_prompt"] == "TARGET_SYS"
    assert v2["max_new_tokens"] == 111
    assert v2["stages"] == 2

    adv = out["runs"]["adversary"]
    assert adv["attacker_instruction"] == "SEED_ATTACKER"
    assert adv["target_system_prompt"] == "TARGET_SYS"
    assert adv["max_new_tokens"] == 111
    assert adv["target_max_new_tokens"] == 222

    vsb = out["runs"]["vector_steering_baseline"]
    assert vsb["seed_prompt"] == "TARGET_SYS"
    assert vsb["max_new_tokens"] == 111
    assert vsb["temperature"] == 0.0
    assert vsb["target_layers"] == [15]


def test_load_default_config_applies_shared(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        """
shared_generation:
  seed_attacker_instruction: "X"
  target_system_prompt: "Y"
  sampling:
    adversary_max_new_tokens: 10
    target_max_new_tokens: 11
    target_temperature: 0.0
    gepa_max_tokens: 12
    gepa_temperature: 0.1
  eval_thresholds:
    refusal_threshold: 0.7
    asr_threshold: 0.3
runs:
  gepa:
    train_size: 1
""",
        encoding="utf-8",
    )
    cfg = load_default_config(path)
    assert cfg["runs"]["gepa"]["baseline_system_prompt"] == "Y"
    assert cfg["runs"]["gepa"]["max_tokens"] == 12
    assert cfg["runs"]["gepa"]["train_size"] == 1


def test_missing_shared_generation_noop() -> None:
    payload = {"runs": {"gepa": {"foo": 1}}}
    assert apply_shared_generation_defaults(payload) == payload

