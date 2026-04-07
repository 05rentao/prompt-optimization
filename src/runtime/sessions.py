"""Runtime session construction: catalog, YAML factories, and timed target helpers."""

from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Literal

from .contracts import (
    HarmbenchJudgeConfig,
    LocalHFConfig,
    OpenAIReflectionConfig,
    OpenAITargetConfig,
    GenerationRequest,
    GenerationSession,
    TargetModelConfig,
    UnslothAdversaryConfig,
)
from .local_runtimes import HarmbenchJudgeRuntime, LocalHFChatRuntime, UnslothAdversaryRuntime
from .openai_http import OpenAIChatTargetRuntime, OpenAIReflectionGateway


class RuntimeCatalog:
    """Builds typed runtime sessions and reflection gateways."""

    @staticmethod
    def build_target_session(cfg: LocalHFConfig) -> GenerationSession:
        """Create a target-model generation session."""
        return GenerationSession(runtime=LocalHFChatRuntime(cfg))

    @staticmethod
    def build_openai_target_session(cfg: OpenAITargetConfig) -> GenerationSession:
        """Create a target session backed by an OpenAI-compatible HTTP endpoint (e.g. vLLM)."""
        return GenerationSession(runtime=OpenAIChatTargetRuntime(cfg))

    @staticmethod
    def build_adversary_session(cfg: UnslothAdversaryConfig) -> GenerationSession:
        """Create an adversary-model generation session."""
        return GenerationSession(runtime=UnslothAdversaryRuntime(cfg))

    @staticmethod
    def build_judge_session(cfg: HarmbenchJudgeConfig) -> GenerationSession:
        """Create a HarmBench judge session."""
        env = os.environ.get("JUDGE_LOAD_IN_4BIT", "").lower()
        if env in ("0", "false", "no"):
            cfg = replace(cfg, load_in_4bit=False)
        return GenerationSession(runtime=HarmbenchJudgeRuntime(cfg))

    @staticmethod
    def build_reflection_gateway(cfg: OpenAIReflectionConfig) -> OpenAIReflectionGateway:
        """Create a frozen OpenAI-compatible reflection gateway."""
        return OpenAIReflectionGateway(cfg)


def _reflection_urls_from_runtime(runtime: dict[str, Any]) -> tuple[str, str]:
    """Shared REFLECTION_VLLM_* env + YAML ``reflection`` block (single source for HTTP creds)."""
    reflection = runtime["reflection"]
    base_url = os.environ.get("REFLECTION_VLLM_BASE_URL", reflection["base_url"])
    api_key = os.environ.get("REFLECTION_VLLM_API_KEY", reflection["api_key"])
    return base_url, api_key


def resolve_reflection_env_overrides(defaults: dict[str, Any]) -> tuple[str, str]:
    """Apply REFLECTION_VLLM_* env overrides over YAML ``runtime.reflection``."""
    return _reflection_urls_from_runtime(defaults["runtime"])


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


def build_vllm_stack(defaults: dict[str, Any]) -> tuple[GenerationSession, OpenAIReflectionGateway]:
    """HTTP target session + reflection gateway from the same YAML and env overrides.

    Pairs :func:`build_vllm_target_session` with :func:`build_reflection_gateway_for_defaults`
    so vLLM wiring stays consistent across run scripts.
    """
    return build_vllm_target_session(defaults), build_reflection_gateway_for_defaults(defaults)


def patch_run_args_from_config(
    defaults: dict[str, Any],
    args: Any,
    *,
    run: Literal["gepa", "coev_v2", "adversary"],
) -> None:
    """Set ``runtime_profile``, model id fields, and reflection URL/key on ``args`` from merged YAML + env."""
    args.runtime_profile = defaults["global"]["runtime_profile"]
    models = defaults["runtime"]["models"]
    rw_url, rw_key = resolve_reflection_env_overrides(defaults)
    args.reflection_vllm_base_url = rw_url
    args.reflection_vllm_api_key = rw_key
    if run == "gepa":
        args.target_model_name = models["target_model_name"]
        args.reflection_model_name = models["reflection_model_name"]
    elif run == "coev_v2":
        args.adversary_model_id = models["adversary_model_id"]
        args.adversary_train_model_id = models.get("adversary_train_model_id", models["adversary_model_id"])
        args.task_model_name = models["target_model_name"]
        args.reflection_model_name = models["reflection_model_name"]
    elif run == "adversary":
        args.adversary_model_id = models["adversary_model_id"]
        args.adversary_train_model_id = models.get("adversary_train_model_id", models["adversary_model_id"])
        args.task_model_name = models["target_model_name"]
        args.judge_model_id = models["judge_model_id"]
    else:
        raise ValueError(f"unknown run patch: {run!r}")


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
    base_url, api_key = _reflection_urls_from_runtime(runtime_defaults)
    return RuntimeCatalog.build_openai_target_session(
        OpenAITargetConfig(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
        )
    )


def timed_target_generate(
    target_session: GenerationSession,
    device: str,
    request: GenerationRequest,
) -> tuple[str, float]:
    """Single target completion with wall-clock latency in milliseconds.

    Safe to call from worker threads when ``target_session.runtime`` exposes
    ``supports_concurrent_target_inference`` (e.g. OpenAI-compatible HTTP targets).
    """
    start = time.perf_counter()
    output = target_session.generate(request, device=device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, elapsed_ms


def cap_thread_workers(
    num_items: int,
    target_max_workers: int | None,
    *,
    default_cap: int = 32,
) -> int:
    """Cap thread-pool size for target calls: ``min(max_workers, num_items)``, at least 1."""
    if num_items <= 0:
        return 1
    workers = target_max_workers if target_max_workers is not None else min(default_cap, max(1, num_items))
    return max(1, min(workers, num_items))


def run_target_requests_ordered(
    target_session: GenerationSession,
    device: str,
    requests: list[GenerationRequest],
    target_max_workers: int | None,
) -> list[tuple[str, float]]:
    """Run target completions in **input order**.

    Sequential when the runtime does not support concurrent inference; otherwise
    submits each request on a thread pool and collects futures in submission order.
    """
    n = len(requests)
    workers = cap_thread_workers(n, target_max_workers)
    use_pool = getattr(target_session.runtime, "supports_concurrent_target_inference", False)

    if not use_pool:
        return [timed_target_generate(target_session, device, req) for req in requests]

    pending: list[Future[tuple[str, float]]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for req in requests:
            fut = executor.submit(timed_target_generate, target_session, device, req)
            pending.append(fut)
        return [fut.result() for fut in pending]
