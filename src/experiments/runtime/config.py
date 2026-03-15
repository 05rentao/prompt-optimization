"""Typed runtime configuration objects used by runtime catalog builders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalHFConfig:
    """Configuration for a local Transformers chat runtime."""

    model_id: str
    use_4bit: bool = True
    max_new_tokens: int = 150


@dataclass
class UnslothAdversaryConfig:
    """Configuration for a local Unsloth adversary runtime with LoRA."""

    model_id: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0


@dataclass
class HarmbenchJudgeConfig:
    """Configuration for HarmBench judge model loading."""

    model_id: str = "cais/HarmBench-Mistral-7b-val-cls"


@dataclass
class OpenAIReflectionConfig:
    """Configuration for OpenAI-compatible frozen reflection endpoint."""

    base_url: str
    api_key: str

