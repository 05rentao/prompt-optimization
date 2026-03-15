"""Shared environment helpers for runtime modules and scripts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Mapping


def resolve_hf_token() -> str:
    """Resolve HF token from standard environment variable names."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    return token


@contextmanager
def scoped_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Temporarily apply environment variable overrides."""
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

