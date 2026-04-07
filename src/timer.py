"""Lightweight section-level timing profiler for training loops."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class Timer:
    """Accumulates wall-clock time per labeled section via context managers.

    Usage::

        timer = Timer()
        with timer.section("rl_update"):
            ...
        with timer.section("gepa_step"):
            ...
        timer.write_json(Path("runs/timing_profile.json"))
    """

    def __init__(self) -> None:
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def section(self, label: str) -> Generator[None, None, None]:
        """Context manager that records elapsed seconds for *label*."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._totals[label] += time.perf_counter() - t0
            self._counts[label] += 1

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        """Return per-label stats: total_s, mean_s, call_count."""
        return {
            label: {
                "total_s": round(self._totals[label], 4),
                "mean_s": round(self._totals[label] / self._counts[label], 4),
                "call_count": self._counts[label],
            }
            for label in self._totals
        }

    def write_json(self, path: Path | str) -> None:
        """Serialize stats to *path*, creating parent directories as needed."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2))
