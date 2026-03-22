"""OpenAI-compatible HTTP target runtime (same vLLM server as GEPA reflection)."""

from __future__ import annotations

from openai import OpenAI

from .config import OpenAITargetConfig
from .interfaces import GenerationRequest, TargetRuntime
from .openai_chat import openai_chat_completion


class OpenAIChatTargetRuntime(TargetRuntime):
    """Target generation via chat.completions against a served model (e.g. vLLM)."""

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
