"""Runtime dataclasses, protocols, and session wrapper used by backends and factories."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# --- Dataclasses (runtime config) ---


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
    #: If set, used instead of the default HarmBench Mistral classifier template. Must include
    #: ``{behavior}`` and ``{generation}`` placeholders for :meth:`HarmbenchJudgeRuntime.judge`.
    classification_prompt_template: str | None = None


@dataclass
class OpenAIReflectionConfig:
    """Configuration for OpenAI-compatible frozen reflection endpoint."""

    base_url: str
    api_key: str


@dataclass
class OpenAITargetConfig:
    """OpenAI-compatible target inference (typically same vLLM URL as reflection)."""

    base_url: str
    api_key: str
    #: Served model id (must match vLLM ``--served-model-name``).
    model_id: str


# --- Protocols and session wrapper ---


@dataclass
class GenerationRequest:
    """Normalized request payload for single-turn chat generation."""

    system_prompt: str
    user_prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.0
    top_p: float = 0.9


@runtime_checkable
class TargetRuntime(Protocol):
    """Protocol for target-model inference backends."""

    def generate(self, request: GenerationRequest, device: str) -> str:
        """Generate a response from a normalized request."""
        ...


@runtime_checkable
class JudgeRuntime(Protocol):
    """Protocol for judge-model backends returning verdict strings."""

    def judge(self, behaviors: str | list[str], generations: list[str]) -> list[str]:
        """Score generations against behavior prompts."""
        ...


@runtime_checkable
class ReflectionGateway(Protocol):
    """Protocol for frozen reflection-model endpoints."""

    def verify(self, reflection_model_name: str) -> None:
        """Validate endpoint reachability and model availability."""
        ...

    def smoke_test(self, reflection_model_name: str) -> str:
        """Run a small completion to confirm endpoint health."""
        ...


@runtime_checkable
class LoRABridge(Protocol):
    """Capability protocol for runtimes that persist LoRA adapters."""

    def save_adapters(self, save_dir: Path) -> None:
        """Save adapter artifacts to a directory."""
        ...


@dataclass
class GenerationSession:
    """Container for a long-lived runtime instance."""

    runtime: Any

    def generate(self, request: GenerationRequest, device: str) -> str:
        """Checks that generations match behaviors and forward generation requests to the bound runtime."""
        if not hasattr(self.runtime, "generate"):
            raise TypeError("Bound runtime does not support generate().")
        return self.runtime.generate(request, device=device)

    def generate_many(
        self,
        requests: list[GenerationRequest],
        device: str,
        *,
        max_workers: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """Generate multiple completions; uses ``runtime.generate_many`` when available, else sequential."""
        if hasattr(self.runtime, "generate_many"):
            return self.runtime.generate_many(requests, device=device, max_workers=max_workers)
        texts: list[str] = []
        latencies: list[float] = []
        for req in requests:
            t0 = time.perf_counter()
            texts.append(self.generate(req, device))
            latencies.append((time.perf_counter() - t0) * 1000.0)
        return texts, latencies

    def judge(self, behaviors: str | list[str], generations: list[str]) -> list[str]:
        """Checks that generations match behaviors and forward judge requests to the bound runtime."""
        if not hasattr(self.runtime, "judge"):
            raise TypeError("Bound runtime does not support judge().")
        return self.runtime.judge(behaviors=behaviors, generations=generations)
