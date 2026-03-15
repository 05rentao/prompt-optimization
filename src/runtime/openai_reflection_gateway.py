"""OpenAI-compatible reflection endpoint gateway for GEPA flows."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from openai import OpenAI

from .config import OpenAIReflectionConfig
from .env import scoped_env
from .interfaces import ReflectionGateway


class OpenAIReflectionGateway(ReflectionGateway):
    """Owns endpoint client, verification, and temporary env binding."""

    def __init__(self, cfg: OpenAIReflectionConfig) -> None:
        """Create a client for an OpenAI-compatible base URL."""
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def list_models(self) -> list[str]:
        """List model IDs exposed by the reflection endpoint."""
        models = self.client.models.list()
        return [model.id for model in models.data]

    def verify(self, reflection_model_name: str) -> None:
        """Validate endpoint reachability and required model presence."""
        try:
            reflection_models = self.list_models()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach reflection vLLM endpoint at {self.cfg.base_url}. "
                "Start an OpenAI-compatible vLLM server and retry."
            ) from exc
        if reflection_model_name not in reflection_models:
            raise RuntimeError(
                f"Reflection model {reflection_model_name!r} is not available on "
                f"{self.cfg.base_url}. Available models (first 20): {reflection_models[:20]}"
            )

    def smoke_test(self, reflection_model_name: str) -> str:
        """Execute a short completion request to verify runtime behavior."""
        response = self.client.chat.completions.create(
            model=reflection_model_name,
            messages=[
                {"role": "system", "content": "You are a prompt optimizer."},
                {"role": "user", "content": "Rewrite: Be safe."},
            ],
            max_tokens=32,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

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

