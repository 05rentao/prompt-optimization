"""Timed target generation and optional concurrent HTTP batches for run scripts."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor

from .interfaces import GenerationRequest, GenerationSession


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
