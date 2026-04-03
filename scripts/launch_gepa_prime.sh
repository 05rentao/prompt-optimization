#!/usr/bin/env bash
set -euo pipefail

# Prime launcher for runs/gepa_run.py
#
# What this script does:
# 1) Ensures uv environment is available
# 2) Starts one vLLM OpenAI server: GEPA reflection and target generation both use it
# 3) Optionally starts a second vLLM on TASK_PORT (START_TASK_VLLM=1) for experiments only
# 4) Runs runs/gepa_run.py (hyperparameters from configs/default.yaml or PROMPT_OPT_CONFIG_PATH)
#
# Important:
# - Target uses the same vLLM as reflection (HTTP); runtime.models in YAML must match REFLECTION_MODEL.
# - This launcher sets REFLECTION_VLLM_BASE_URL to match REFLECTION_PORT (default 8001).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:."

TASK_PORT="${TASK_PORT:-8000}"
REFLECTION_PORT="${REFLECTION_PORT:-8001}"

TASK_MODEL="${TASK_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
REFLECTION_MODEL="${REFLECTION_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

TASK_GPU_UTIL="${TASK_GPU_UTIL:-0.35}"
REFLECTION_GPU_UTIL="${REFLECTION_GPU_UTIL:-0.35}"

TASK_MAX_MODEL_LEN="${TASK_MAX_MODEL_LEN:-4096}"
REFLECTION_MAX_MODEL_LEN="${REFLECTION_MAX_MODEL_LEN:-8192}"

# Second vLLM on TASK_PORT is optional (legacy / A/B); normal runs use a single server.
START_TASK_VLLM="${START_TASK_VLLM:-0}"

KEEP_VLLM_UP="${KEEP_VLLM_UP:-0}"

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

port_is_open() {
  local port="$1"
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "${port}" 2>/dev/null
    return $?
  fi
  (exec 3<>/dev/tcp/127.0.0.1/"${port}") 2>/dev/null
}

wait_for_port() {
  local port="$1"
  local name="$2"
  local timeout_s="${3:-180}"
  local waited=0
  echo "Waiting for ${name} on :${port} (timeout ${timeout_s}s)..."
  until port_is_open "${port}"; do
    sleep 2
    waited=$((waited + 2))
    if [[ "${waited}" -ge "${timeout_s}" ]]; then
      echo "ERROR: ${name} did not become ready on :${port} in time."
      return 1
    fi
  done
  echo "${name} is up on :${port}"
}

cleanup() {
  if [[ "${KEEP_VLLM_UP}" == "1" ]]; then
    echo "KEEP_VLLM_UP=1 set; leaving vLLM servers running."
    return
  fi
  echo "Cleaning up vLLM processes..."
  if [[ -n "${TASK_VLLM_PID:-}" ]]; then
    kill "${TASK_VLLM_PID}" 2>/dev/null || true
  fi
  if [[ -n "${REFLECTION_VLLM_PID:-}" ]]; then
    kill "${REFLECTION_VLLM_PID}" 2>/dev/null || true
  fi
  fuser -k "${TASK_PORT}/tcp" "${REFLECTION_PORT}/tcp" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Ensuring target ports are free..."
fuser -k "${TASK_PORT}/tcp" "${REFLECTION_PORT}/tcp" >/dev/null 2>&1 || true

if [[ "${START_TASK_VLLM}" == "1" ]]; then
  echo "Starting optional task vLLM (${TASK_MODEL}) on :${TASK_PORT}..."
  uv run python -m vllm.entrypoints.openai.api_server \
    --model "${TASK_MODEL}" \
    --served-model-name "${TASK_MODEL}" \
    --host 0.0.0.0 \
    --port "${TASK_PORT}" \
    --max-model-len "${TASK_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${TASK_GPU_UTIL}" \
    --enforce-eager > logs/gepa_prime_task_vllm.log 2>&1 &
  TASK_VLLM_PID=$!
  wait_for_port "${TASK_PORT}" "task vLLM" 240
else
  echo "Skipping task vLLM (START_TASK_VLLM=0)."
fi

echo "Starting reflection vLLM (${REFLECTION_MODEL}) on :${REFLECTION_PORT}..."
uv run python -m vllm.entrypoints.openai.api_server \
  --model "${REFLECTION_MODEL}" \
  --served-model-name "${REFLECTION_MODEL}" \
  --host 0.0.0.0 \
  --port "${REFLECTION_PORT}" \
  --max-model-len "${REFLECTION_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${REFLECTION_GPU_UTIL}" \
  --enforce-eager > logs/gepa_prime_reflection_vllm.log 2>&1 &
REFLECTION_VLLM_PID=$!
wait_for_port "${REFLECTION_PORT}" "reflection vLLM" 240

export REFLECTION_VLLM_BASE_URL="http://127.0.0.1:${REFLECTION_PORT}/v1"
export REFLECTION_VLLM_API_KEY="${REFLECTION_VLLM_API_KEY:-EMPTY}"
echo "REFLECTION_VLLM_BASE_URL=${REFLECTION_VLLM_BASE_URL}"

echo "Launching runs/gepa_run.py..."
RUN_CMD=(uv run python runs/gepa_run.py)

if [[ -n "${RESULTS_DIR:-}" ]]; then
  RUN_CMD+=(--results-dir "${RESULTS_DIR}")
fi
if [[ -n "${BASELINE_SYSTEM_PROMPT:-}" ]]; then
  RUN_CMD+=(--baseline-system-prompt "${BASELINE_SYSTEM_PROMPT}")
fi

"${RUN_CMD[@]}"

echo "Run complete."
echo "Reflection vLLM log: logs/gepa_prime_reflection_vllm.log"
if [[ "${START_TASK_VLLM}" == "1" ]]; then
  echo "Task vLLM log: logs/gepa_prime_task_vllm.log"
fi
echo "Artifacts: see runs.gepa.results_dir in configs/default.yaml (default: results)."
