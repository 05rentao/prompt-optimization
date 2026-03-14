#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for a single H100-80GB Prime instance.
# Modes:
#   - mark:   start dual vLLM endpoints, then run Mark GEPA path
#   - coev:   run local CoEV path only (no vLLM endpoints)
#   - hybrid: start dual vLLM endpoints, run mark then coev via unified runner

MODE="${MODE:-mark}"                         # mark | coev | hybrid
RUNTIME_PROFILE="${RUNTIME_PROFILE:-dual_vllm}"
HYBRID_ORDER="${HYBRID_ORDER:-mark_then_coev}"

TASK_PORT="${TASK_PORT:-8000}"
REFLECTION_PORT="${REFLECTION_PORT:-8001}"

TASK_MODEL="${TASK_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
REFLECTION_MODEL="${REFLECTION_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# Conservative split for one H100.
TASK_GPU_UTIL="${TASK_GPU_UTIL:-0.40}"
REFLECTION_GPU_UTIL="${REFLECTION_GPU_UTIL:-0.40}"
TASK_MAX_MODEL_LEN="${TASK_MAX_MODEL_LEN:-4096}"
REFLECTION_MAX_MODEL_LEN="${REFLECTION_MAX_MODEL_LEN:-8192}"

DATASET_NAME="${DATASET_NAME:-walledai/HarmBench}"
DATASET_CONFIG="${DATASET_CONFIG:-standard}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TRAIN_SIZE="${TRAIN_SIZE:-100}"
VAL_SIZE="${VAL_SIZE:-100}"
SEED="${SEED:-42}"

MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-300}"
COEV_MODE="${COEV_MODE:-reinforce}"          # reinforce | gepa | eval

MARK_RESULTS_DIR="${MARK_RESULTS_DIR:-results/mark}"
COEV_RESULTS_DIR="${COEV_RESULTS_DIR:-results/coev}"

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
  echo "WARNING: HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set. Gated models/datasets may fail."
fi

cleanup_vllm() {
  pkill -f "vllm.entrypoints.openai.api_server" || true
  fuser -k "${TASK_PORT}/tcp" "${REFLECTION_PORT}/tcp" || true
}

wait_for_port() {
  local port="$1"
  local name="$2"
  echo "Waiting for ${name} on port ${port}..."
  until nc -z 127.0.0.1 "${port}"; do
    sleep 2
  done
  echo "${name} is up on :${port}"
}

run_unified() {
  uv run python scripts/run_unified_experiment.py \
    --mode "${MODE}" \
    --hybrid-order "${HYBRID_ORDER}" \
    --runtime-profile "${RUNTIME_PROFILE}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-config "${DATASET_CONFIG}" \
    --dataset-split "${DATASET_SPLIT}" \
    --train-size "${TRAIN_SIZE}" \
    --val-size "${VAL_SIZE}" \
    --seed "${SEED}" \
    --task-model-name "${TASK_MODEL}" \
    --reflection-model-name "${REFLECTION_MODEL}" \
    --vllm-base-url "http://127.0.0.1:${TASK_PORT}/v1" \
    --reflection-vllm-base-url "http://127.0.0.1:${REFLECTION_PORT}/v1" \
    --max-metric-calls "${MAX_METRIC_CALLS}" \
    --coev-mode "${COEV_MODE}" \
    --mark-results-dir "${MARK_RESULTS_DIR}" \
    --coev-results-dir "${COEV_RESULTS_DIR}"
}

if [[ "${MODE}" == "coev" ]]; then
  echo "Running CoEV only (no vLLM servers)..."
  run_unified
  echo "Done."
  exit 0
fi

echo "Starting dual-vLLM runtime for mode=${MODE}..."
cleanup_vllm

uv run python -m vllm.entrypoints.openai.api_server \
  --model "${TASK_MODEL}" \
  --served-model-name "${TASK_MODEL}" \
  --host 0.0.0.0 \
  --port "${TASK_PORT}" \
  --max-model-len "${TASK_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${TASK_GPU_UTIL}" \
  --enforce-eager > logs/unified_task_vllm.log 2>&1 &
wait_for_port "${TASK_PORT}" "task vLLM"

uv run python -m vllm.entrypoints.openai.api_server \
  --model "${REFLECTION_MODEL}" \
  --served-model-name "${REFLECTION_MODEL}" \
  --host 0.0.0.0 \
  --port "${REFLECTION_PORT}" \
  --max-model-len "${REFLECTION_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${REFLECTION_GPU_UTIL}" \
  --enforce-eager > logs/unified_reflection_vllm.log 2>&1 &
wait_for_port "${REFLECTION_PORT}" "reflection vLLM"

run_unified

echo "Run complete."
echo "Logs: logs/unified_task_vllm.log, logs/unified_reflection_vllm.log"
echo "Artifacts: ${MARK_RESULTS_DIR}, ${COEV_RESULTS_DIR}"
