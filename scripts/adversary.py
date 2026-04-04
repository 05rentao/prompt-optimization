#!/usr/bin/env python3
"""Run single-shot adversary experiments (delegates to ``runs/adversary_run.py``).

From the repo root::

    uv run python scripts/adversary.py --mode train --no-finetune --adversary-prompt decompose \\
      --results-dir results/adversary_decompose

**Artifacts** (under ``--results-dir``) include rewriter/target prompts in:

- ``adversary_run_metrics.json`` (``prompts`` key)
- ``prompts.json`` (same payload, standalone for tooling)
- ``run_manifest.json`` (``extra.prompts`` and ``extra.prompts_json_path``)

All other flags match ``runs/adversary_run.py`` (see ``--help`` there).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADVERSARY_RUN = REPO_ROOT / "runs" / "adversary_run.py"


def main() -> None:
    cmd = [sys.executable, str(ADVERSARY_RUN), *sys.argv[1:]]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
