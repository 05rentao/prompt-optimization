"""Shared runtime interfaces for model generation and reflection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


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

    def judge(self, behaviors: str | list[str], generations: list[str]) -> list[str]:
        """Checks that generations match behaviors and forward judge requests to the bound runtime."""
        if not hasattr(self.runtime, "judge"):
            raise TypeError("Bound runtime does not support judge().")
        return self.runtime.judge(behaviors=behaviors, generations=generations)

