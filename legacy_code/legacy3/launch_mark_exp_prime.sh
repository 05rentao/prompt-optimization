#!/usr/bin/env bash
set -euo pipefail

# Prime launcher for runs/gepa_run.py
# Starts two local vLLM OpenAI-compatible endpoints:
#   - Task model on :8000
#   - Reflection model on :8001
# Then runs GEPA optimization via uv.

# ----------------------------
# User-configurable defaults
# ----------------------------
TASK_PORT="${TASK_PORT:-8000}"
REFLECTION_PORT="${REFLECTION_PORT:-8001}"

TASK_MODEL="${TASK_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
REFLECTION_MODEL="${REFLECTION_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

TASK_GPU_UTIL="${TASK_GPU_UTIL:-0.45}"
REFLECTION_GPU_UTIL="${REFLECTION_GPU_UTIL:-0.45}"

TASK_MAX_MODEL_LEN="${TASK_MAX_MODEL_LEN:-4096}"
REFLECTION_MAX_MODEL_LEN="${REFLECTION_MAX_MODEL_LEN:-8192}"

DATASET_NAME="${DATASET_NAME:-walledai/HarmBench}"
DATASET_CONFIG="${DATASET_CONFIG:-standard}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TRAIN_SIZE="${TRAIN_SIZE:-100}"
VAL_SIZE="${VAL_SIZE:-100}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-300}"

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

cleanup() {
  echo "Cleaning up vLLM processes/ports..."
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

cleanup

echo "Starting task vLLM (${TASK_MODEL}) on :${TASK_PORT}..."
uv run python -m vllm.entrypoints.openai.api_server \
  --model "${TASK_MODEL}" \
  --served-model-name "${TASK_MODEL}" \
  --host 0.0.0.0 \
  --port "${TASK_PORT}" \
  --max-model-len "${TASK_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${TASK_GPU_UTIL}" \
  --enforce-eager > logs/mark_exp_task_vllm.log 2>&1 &
wait_for_port "${TASK_PORT}" "task vLLM"

echo "Starting reflection vLLM (${REFLECTION_MODEL}) on :${REFLECTION_PORT}..."
uv run python -m vllm.entrypoints.openai.api_server \
  --model "${REFLECTION_MODEL}" \
  --served-model-name "${REFLECTION_MODEL}" \
  --host 0.0.0.0 \
  --port "${REFLECTION_PORT}" \
  --max-model-len "${REFLECTION_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${REFLECTION_GPU_UTIL}" \
  --enforce-eager > logs/mark_exp_reflection_vllm.log 2>&1 &
wait_for_port "${REFLECTION_PORT}" "reflection vLLM"

echo "Launching gepa_run.py with dual local vLLM endpoints..."
uv run python runs/gepa_run.py \
  --vllm-base-url "http://127.0.0.1:${TASK_PORT}/v1" \
  --task-model-name "${TASK_MODEL}" \
  --reflection-vllm-base-url "http://127.0.0.1:${REFLECTION_PORT}/v1" \
  --reflection-model-name "${REFLECTION_MODEL}" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-config "${DATASET_CONFIG}" \
  --dataset-split "${DATASET_SPLIT}" \
  --train-size "${TRAIN_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --max-metric-calls "${MAX_METRIC_CALLS}" \
  --show-progress

echo "Run complete."
echo "Logs: logs/mark_exp_task_vllm.log, logs/mark_exp_reflection_vllm.log"
echo "Artifacts: optimized_system_prompt.txt, gepa_run_metrics.json, results/"
