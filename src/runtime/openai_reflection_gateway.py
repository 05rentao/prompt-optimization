"""OpenAI-compatible reflection endpoint gateway for GEPA flows."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator

import requests
from openai import OpenAI

from .config import OpenAIReflectionConfig
from .env import scoped_env
from .interfaces import ReflectionGateway


def _response_looks_like_html(r: requests.Response) -> bool:
    ct = (r.headers.get("Content-Type") or "").lower()
    if "text/html" in ct:
        return True
    t = (r.text or "").lstrip()
    return t.startswith("<!DOCTYPE") or t.startswith("<html") or t.startswith("<HTML")


class OpenAIReflectionGateway(ReflectionGateway):
    """Owns endpoint client, verification, and temporary env binding."""

    def __init__(self, cfg: OpenAIReflectionConfig) -> None:
        """Create a client for an OpenAI-compatible base URL."""
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def list_models(self) -> list[str]:
        """List model IDs exposed by the reflection endpoint.

        Uses HTTP + JSON instead of ``client.models.list()``: OpenAI Python SDK 2.x
        builds a typed ``SyncPage[Model]`` from the response; some OpenAI-compatible
        servers return JSON whose root is a JSON string (double-encoded) or ``data``
        entries as plain model-id strings. That yields a ``str`` at parse time and
        breaks the SDK (``'str' object has no attribute '_set_private_attributes'``).

        Retries are needed because the launcher only waits for TCP on the port; vLLM may
        accept connections before ``GET /v1/models`` returns non-empty JSON (empty body
        until the OpenAI app is fully ready).
        """
        url = f"{self.cfg.base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        max_attempts = 120
        delay_s = 0.5
        last_detail = "no attempts"
        payload: Any = None
        for attempt in range(max_attempts):
            try:
                r = requests.get(url, headers=headers, timeout=30)
            except requests.RequestException as exc:
                last_detail = f"request error: {exc!r}"
                time.sleep(delay_s)
                continue
            if r.status_code >= 500:
                last_detail = f"HTTP {r.status_code}: {r.text[:300]!r}"
                time.sleep(delay_s)
                continue
            if r.status_code >= 400:
                r.raise_for_status()
            text = (r.text or "").strip()
            if not text:
                last_detail = f"HTTP {r.status_code}, empty body (reflection HTTP not ready yet?)"
                time.sleep(delay_s)
                continue
            if _response_looks_like_html(r):
                raise RuntimeError(
                    f"GET {url} returned HTML (not OpenAI JSON). This often means a host/proxy "
                    f"error (e.g. RunPod 502 page) or the wrong service on the port — not vLLM. "
                    f"Status={r.status_code}, Content-Type={r.headers.get('Content-Type')!r}. "
                    "Fix port forwarding / ensure vLLM is listening, or wait until /v1/models returns JSON."
                )
            try:
                payload = r.json()
            except requests.exceptions.JSONDecodeError:
                last_detail = f"HTTP {r.status_code}, non-JSON body: {text[:300]!r}"
                time.sleep(delay_s)
                continue
            break
        else:
            raise RuntimeError(
                f"Could not read JSON model list from {url} after {max_attempts} attempts "
                f"({max_attempts * delay_s:.0f}s). Last: {last_detail}"
            )
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected /models response from {url}: expected JSON object, got {type(payload).__name__}"
            )
        raw_data = payload.get("data")
        if raw_data is None:
            ids: list[str] = []
        elif isinstance(raw_data, list):
            ids = []
            for item in raw_data:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict):
                    mid = item.get("id")
                    if isinstance(mid, str):
                        ids.append(mid)
        else:
            raise RuntimeError(
                f"Unexpected /models response from {url}: 'data' must be a list or null, "
                f"got {type(raw_data).__name__}"
            )
        return ids

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

