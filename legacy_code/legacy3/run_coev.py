"""Compatibility shim for legacy CoEV path.

Primary implementation now lives at `runs/coev_run.py`.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runs.coev_run import main


if __name__ == "__main__":
    main()
