#!/usr/bin/env bash
set -euo pipefail

# Prime launcher for runs/vector_steering_baseline.py
#
# What this script does:
# 1) Ensures uv environment is available
# 2) Syncs project dependencies
# 3) Runs vector steering baseline with configurable flags

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TRAIN_SIZE="${TRAIN_SIZE:-100}"
VAL_SIZE="${VAL_SIZE:-100}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-}"

TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-cais/HarmBench-Mistral-7b-val-cls}"
SEED_PROMPT="${SEED_PROMPT:-You are a helpful assistant.}"

CAA_DATA_PATH="${CAA_DATA_PATH:-data/refusal_CAA_training.json}"
TARGET_LAYERS="${TARGET_LAYERS:-15}"
STEERING_COEFFICIENT="${STEERING_COEFFICIENT:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"

RESULTS_DIR="${RESULTS_DIR:-results/vector_steering_baseline_prime}"
VECTOR_DIR="${VECTOR_DIR:-vectors}"
RESULTS_CSV_NAME="${RESULTS_CSV_NAME:-steered_eval_outputs.csv}"
SUMMARY_JSON_NAME="${SUMMARY_JSON_NAME:-vector_steering_summary.json}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

mkdir -p logs results outputs data "${RESULTS_DIR}"

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

read -r -a TARGET_LAYER_ARGS <<< "${TARGET_LAYERS}"

echo "Launching runs/vector_steering_baseline.py..."
RUN_CMD=(
  uv run python runs/vector_steering_baseline.py
  --train-size "${TRAIN_SIZE}"
  --val-size "${VAL_SIZE}"
  --seed "${SEED}"
  --target-model-name "${TARGET_MODEL_NAME}"
  --judge-model-id "${JUDGE_MODEL_ID}"
  --seed-prompt "${SEED_PROMPT}"
  --caa-data-path "${CAA_DATA_PATH}"
  --target-layers "${TARGET_LAYER_ARGS[@]}"
  --steering-coefficient "${STEERING_COEFFICIENT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --results-dir "${RESULTS_DIR}"
  --vector-dir "${VECTOR_DIR}"
  --results-csv-name "${RESULTS_CSV_NAME}"
  --summary-json-name "${SUMMARY_JSON_NAME}"
)

if [[ -n "${DEVICE}" ]]; then
  RUN_CMD+=(--device "${DEVICE}")
fi

if [[ "${SHOW_PROGRESS}" == "1" ]]; then
  RUN_CMD+=(--show-progress)
fi

"${RUN_CMD[@]}"

echo "Run complete."
echo "Artifacts: ${RESULTS_DIR}"
