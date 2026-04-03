"""OpenAI-compatible HTTP clients: target completions and reflection gateway (shared vLLM server)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, ClassVar, Iterator

from openai import OpenAI

from .contracts import (
    GenerationRequest,
    OpenAIReflectionConfig,
    OpenAITargetConfig,
    ReflectionGateway,
    TargetRuntime,
)
from .defaults import scoped_env


def openai_chat_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Run one chat completion and return assistant text content."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if temperature > 0.0:
        kwargs["top_p"] = top_p
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


class OpenAIChatTargetRuntime(TargetRuntime):
    """Target generation via chat.completions against a served model (e.g. vLLM)."""

    supports_concurrent_target_inference: ClassVar[bool] = True

    def __init__(self, cfg: OpenAITargetConfig) -> None:
        self._cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def generate(self, request: GenerationRequest, device: str) -> str:
        """Map a normalized request to OpenAI chat; ``device`` is ignored (no local weights)."""
        _ = device
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        return openai_chat_completion(
            self._client,
            model=self._cfg.model_id,
            messages=messages,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

    def generate_many(
        self,
        requests: list[GenerationRequest],
        device: str,
        *,
        max_workers: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """Run multiple chat completions concurrently (HTTP); preserves request order."""
        _ = device
        if not requests:
            return [], []
        if len(requests) == 1:
            t0 = time.perf_counter()
            text = self.generate(requests[0], device)
            ms = (time.perf_counter() - t0) * 1000.0
            return [text], [ms]

        n = len(requests)
        workers = max_workers if max_workers is not None else min(32, n)
        workers = max(1, min(workers, n))
        if workers == 1:
            texts: list[str] = []
            latencies: list[float] = []
            for req in requests:
                t0 = time.perf_counter()
                texts.append(self.generate(req, device))
                latencies.append((time.perf_counter() - t0) * 1000.0)
            return texts, latencies

        def _one(req: GenerationRequest) -> tuple[str, float]:
            t0 = time.perf_counter()
            out = self.generate(req, device)
            return out, (time.perf_counter() - t0) * 1000.0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_one, requests))
        return [r[0] for r in results], [r[1] for r in results]


class OpenAIReflectionGateway(ReflectionGateway):
    """Owns endpoint client, verification, and temporary env binding."""

    def __init__(self, cfg: OpenAIReflectionConfig) -> None:
        """Create a client for an OpenAI-compatible base URL."""
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def verify(self, reflection_model_name: str) -> None:
        """Validate the OpenAI-compatible server with one chat completion on ``reflection_model_name``.

        Assumes a single local vLLM (or shared target+reflection) with that model id served
        — no ``GET /v1/models`` catalog. Retries cover the common case where TCP is open
        before the chat API is ready.
        """
        max_attempts = 120
        delay_s = 0.5
        last_exc: Exception | None = None
        for _ in range(max_attempts):
            try:
                openai_chat_completion(
                    self.client,
                    model=reflection_model_name,
                    messages=[{"role": "user", "content": "."}],
                    max_tokens=1,
                    temperature=0.0,
                    top_p=0.9,
                )
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(delay_s)
        raise RuntimeError(
            f"Cannot complete with reflection model {reflection_model_name!r} at {self.cfg.base_url} "
            f"after {max_attempts} attempts (~{max_attempts * delay_s:.0f}s). "
            "Ensure vLLM is up and --served-model-name matches the config."
        ) from last_exc

    def smoke_test(self, reflection_model_name: str) -> str:
        """Execute a short completion request to verify runtime behavior."""
        return openai_chat_completion(
            self.client,
            model=reflection_model_name,
            messages=[
                {"role": "system", "content": "You are a prompt optimizer."},
                {"role": "user", "content": "Rewrite: Be safe."},
            ],
            max_tokens=32,
            temperature=0.0,
            top_p=0.9,
        )

    @contextmanager
    def bind_openai_env(self) -> Iterator[None]:
        """Temporarily route global OpenAI env vars to this endpoint."""
        with scoped_env(
            {
                "OPENAI_API_KEY": self.cfg.api_key,
                "OPENAI_BASE_URL": self.cfg.base_url,
                "OPENAI_API_BASE": self.cfg.base_url,
            }
        ):
            yield
