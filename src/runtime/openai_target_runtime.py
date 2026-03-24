"""OpenAI-compatible HTTP target runtime (same vLLM server as GEPA reflection)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

from openai import OpenAI

from .config import OpenAITargetConfig
from .interfaces import GenerationRequest, TargetRuntime
from .openai_chat import openai_chat_completion


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
