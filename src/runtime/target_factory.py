"""Factory helpers for building the default target session (vLLM vs local HF)."""

from __future__ import annotations

import os
import warnings
from typing import Any

from .catalog import RuntimeCatalog
from .config import LocalHFConfig, OpenAIReflectionConfig, OpenAITargetConfig, TargetModelConfig
from .interfaces import GenerationSession
from .openai_reflection_gateway import OpenAIReflectionGateway


def resolve_reflection_env_overrides(defaults: dict[str, Any]) -> tuple[str, str]:
    """Apply REFLECTION_VLLM_* env overrides over YAML ``runtime.reflection``."""
    reflection = defaults["runtime"]["reflection"]
    base_url = os.environ.get("REFLECTION_VLLM_BASE_URL", reflection["base_url"])
    api_key = os.environ.get("REFLECTION_VLLM_API_KEY", reflection["api_key"])
    return base_url, api_key


def build_reflection_gateway_for_defaults(defaults: dict[str, Any]) -> OpenAIReflectionGateway:
    """Create a reflection gateway using YAML + env overrides."""
    base_url, api_key = resolve_reflection_env_overrides(defaults)
    return RuntimeCatalog.build_reflection_gateway(OpenAIReflectionConfig(base_url=base_url, api_key=api_key))


def build_vllm_target_session(defaults: dict[str, Any]) -> GenerationSession:
    """Create a target session that calls the same OpenAI-compatible server as GEPA reflection.

    Uses ``reflection_model_name`` as the served ``model`` id (must match vLLM ``--served-model-name``).
    Warns if ``target_model_name`` differs from ``reflection_model_name``.
    """
    models = defaults["runtime"]["models"]
    target_name = models["target_model_name"]
    reflection_name = models["reflection_model_name"]
    if target_name != reflection_name:
        warnings.warn(
            f"target_model_name ({target_name!r}) != reflection_model_name ({reflection_name!r}); "
            "using reflection_model_name as the OpenAI chat model id.",
            stacklevel=2,
        )
    base_url, api_key = resolve_reflection_env_overrides(defaults)
    return RuntimeCatalog.build_openai_target_session(
        OpenAITargetConfig(base_url=base_url, api_key=api_key, model_id=reflection_name)
    )


def build_local_hf_target_session(cfg: TargetModelConfig) -> GenerationSession:
    """Create a local Transformers target session (vector steering and similar)."""
    return RuntimeCatalog.build_target_session(
        LocalHFConfig(
            model_id=cfg.model_id,
            use_4bit=True,
            max_new_tokens=cfg.max_new_tokens,
        )
    )
