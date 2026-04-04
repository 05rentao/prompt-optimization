"""Tests for adversary rewriter resolution."""

from __future__ import annotations

from src.runtime.adversary_prompts import (
    ADVERSARY_PROMPT_VARIANTS,
    resolve_adversary_attacker_instruction,
)


def test_resolve_cli_wins_over_variant_and_yaml() -> None:
    out = resolve_adversary_attacker_instruction(
        cli_explicit="CUSTOM",
        adversary_prompt="persona",
        merged_yaml_instruction="FROM_YAML",
    )
    assert out == "CUSTOM"


def test_resolve_non_default_variant_over_yaml_seed() -> None:
    out = resolve_adversary_attacker_instruction(
        cli_explicit=None,
        adversary_prompt="persona",
        merged_yaml_instruction="SEED_FROM_SHARED_GENERATION",
    )
    assert "credible professional" in out
    assert out == ADVERSARY_PROMPT_VARIANTS["persona"]


def test_resolve_default_uses_yaml_when_present() -> None:
    out = resolve_adversary_attacker_instruction(
        cli_explicit=None,
        adversary_prompt="default",
        merged_yaml_instruction="YAML_SEED",
    )
    assert out == "YAML_SEED"


def test_resolve_default_falls_back_to_variant() -> None:
    out = resolve_adversary_attacker_instruction(
        cli_explicit=None,
        adversary_prompt="default",
        merged_yaml_instruction=None,
    )
    assert out == ADVERSARY_PROMPT_VARIANTS["default"]
