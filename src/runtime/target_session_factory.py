"""Build target GenerationSession from runtime YAML + env (local HF vs OpenAI/vLLM)."""

from __future__ import annotations

import os
from typing import Literal

from .catalog import RuntimeCatalog
from .config import LocalHFConfig, OpenAITargetConfig


def resolve_target_backend(runtime_defaults: dict, *, force_local_hf: bool = False) -> Literal["openai", "local_hf"]:
    """Resolve backend: env TARGET_BACKEND overrides YAML runtime.target.backend (default openai)."""
    if force_local_hf:
        return "local_hf"
    env = os.environ.get("TARGET_BACKEND", "").strip().lower()
    if env in ("local_hf", "openai"):
        return env  # type: ignore[return-value]
    target_section = runtime_defaults.get("target") or {}
    backend = str(target_section.get("backend", "openai")).strip().lower()
    if backend in ("local_hf", "openai"):
        return backend  # type: ignore[return-value]
    return "openai"


def build_target_session_from_runtime(
    runtime_defaults: dict,
    max_new_tokens: int,
    *,
    force_local_hf: bool = False,
):
    """Construct target session: OpenAI/vLLM by default, local HF for steering or TARGET_BACKEND."""
    models = runtime_defaults["models"]
    model_id = models["target_model_name"]
    backend = resolve_target_backend(runtime_defaults, force_local_hf=force_local_hf)
    if backend == "local_hf":
        return RuntimeCatalog.build_target_session(
            LocalHFConfig(
                model_id=model_id,
                use_4bit=True,
                max_new_tokens=max_new_tokens,
            )
        )
    reflection = runtime_defaults["reflection"]
    base_url = os.environ.get("REFLECTION_VLLM_BASE_URL", reflection["base_url"])
    api_key = os.environ.get("REFLECTION_VLLM_API_KEY", reflection["api_key"])
    return RuntimeCatalog.build_openai_target_session(
        OpenAITargetConfig(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    )
