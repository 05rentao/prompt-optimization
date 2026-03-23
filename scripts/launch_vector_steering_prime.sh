#!/usr/bin/env bash
set -euo pipefail

# Prime launcher for runs/vector_steering_baseline.py
#
# What this script does:
# 1) Ensures uv environment is available
# 2) Syncs project dependencies
# 3) Runs vector steering baseline (hyperparameters from configs/default.yaml or PROMPT_OPT_CONFIG_PATH)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

mkdir -p logs results outputs data

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

echo "Syncing environment with uv..."
uv sync

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set. Gated models may fail."
fi

echo "Launching runs/vector_steering_baseline.py..."
RUN_CMD=(uv run python runs/vector_steering_baseline.py)

if [[ -n "${DEVICE}" ]]; then
  RUN_CMD+=(--device "${DEVICE}")
fi
if [[ -n "${RESULTS_DIR:-}" ]]; then
  RUN_CMD+=(--results-dir "${RESULTS_DIR}")
fi
if [[ -n "${VECTOR_DIR:-}" ]]; then
  RUN_CMD+=(--vector-dir "${VECTOR_DIR}")
fi
if [[ "${SHOW_PROGRESS}" == "1" ]]; then
  RUN_CMD+=(--show-progress)
fi

"${RUN_CMD[@]}"

echo "Run complete."
echo "Artifacts: see runs.vector_steering_baseline in configs/default.yaml (or your PROMPT_OPT_CONFIG_PATH)."
