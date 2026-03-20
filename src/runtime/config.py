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
class TargetModelConfig:
    """Shared target-model config used by experiment runners."""

    model_id: str
    max_new_tokens: int = 150


@dataclass
class ModelConfig:
    """Shared adversary model config used by CoEV-style runners."""

    model_id: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0


@dataclass
class CoevConfig:
    """Shared stage-level CoEV training schedule and optimization knobs."""

    stages: int = 2
    iters_per_stage: int = 5
    eval_every_stages: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.01
    train_slice_end: int = 50
    eval_slice_start: int = 100
    eval_slice_end: int = 110
    csv_path: str = "smoke_test_coev_v2.csv"
    initial_attacker_instruction: str = ""
    initial_defense_prompt: str = ""


@dataclass
class GepaOptimizationConfig:
    """Shared GEPA budget/runtime knobs for coevolution boundary updates."""

    max_metric_calls: int = 100
    max_tokens: int = 120
    temperature: float = 0.0
    reflection_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"


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
    #: NF4 4-bit on GPU (~4GB weights vs ~14GB bf16). Set False only if you need exact bf16 judge.
    load_in_4bit: bool = True


@dataclass
class OpenAIReflectionConfig:
    """Configuration for OpenAI-compatible frozen reflection endpoint."""

    base_url: str
    api_key: str

