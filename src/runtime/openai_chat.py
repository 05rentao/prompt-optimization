"""Shared OpenAI-compatible chat completion helpers for vLLM and similar servers."""

from __future__ import annotations

from typing import Any

from openai import OpenAI


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
