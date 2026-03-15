from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import RunManifest


def write_run_manifest(results_dir: Path, payload: RunManifest | dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "run_manifest.json"

    if isinstance(payload, RunManifest):
        body = payload.to_dict()
    else:
        body = payload

    out_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return out_path
