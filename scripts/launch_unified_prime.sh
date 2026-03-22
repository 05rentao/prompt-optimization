#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:."

# Unified launcher for a single H100-80GB Prime instance.
# Starts one vLLM OpenAI server (REFLECTION_MODEL) used for GEPA reflection and target generation.
# Modes:
#   - gepa:      GEPA prompt optimization
#   - coev:      CoEV reinforce/gepa/eval (target via same vLLM)
#   - coev_v2:   CoEV v2 + dual-role GEPA
#   - adversary: adversary-only training (target via same vLLM)

MODE="${MODE:-gepa}"                         # gepa | coev | coev_v2 | adversary

REFLECTION_PORT="${REFLECTION_PORT:-8001}"

REFLECTION_MODEL="${REFLECTION_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# Conservative split for one H100.
REFLECTION_GPU_UTIL="${REFLECTION_GPU_UTIL:-0.40}"
REFLECTION_MAX_MODEL_LEN="${REFLECTION_MAX_MODEL_LEN:-8192}"

DATASET_NAME="${DATASET_NAME:-walledai/HarmBench}"
DATASET_CONFIG="${DATASET_CONFIG:-standard}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TRAIN_SIZE="${TRAIN_SIZE:-100}"
VAL_SIZE="${VAL_SIZE:-100}"
SEED="${SEED:-42}"

MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-300}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
COEV_MODE="${COEV_MODE:-reinforce}"          # reinforce | gepa | eval
COEV_V2_MODE="${COEV_V2_MODE:-coev}"         # coev | eval
ADVERSARY_MODE="${ADVERSARY_MODE:-train}"    # train | eval
DEVICE="${DEVICE:-}"
EVAL_METHOD="${EVAL_METHOD:-judge}"          # judge | heuristic
REFUSAL_THRESHOLD="${REFUSAL_THRESHOLD:-0.7}"
ASR_THRESHOLD="${ASR_THRESHOLD:-0.3}"
SAVE_DIR="${SAVE_DIR:-}"

GEPA_RESULTS_DIR="${GEPA_RESULTS_DIR:-results/gepa}"
COEV_RESULTS_DIR="${COEV_RESULTS_DIR:-results/coev}"
ADVERSARY_RESULTS_DIR="${ADVERSARY_RESULTS_DIR:-results/adversary}"

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
  fuser -k "${REFLECTION_PORT}/tcp" || true
}

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
  echo "Waiting for ${name} on port ${port}..."
  until port_is_open "${port}"; do
    sleep 2
  done
  echo "${name} is up on :${port}"
}

run_unified() {
  local -a cmd=(
    uv run python scripts/run_unified_experiment.py
    --mode "${MODE}"
    --dataset-name "${DATASET_NAME}"
    --dataset-config "${DATASET_CONFIG}"
    --dataset-split "${DATASET_SPLIT}"
    --train-size "${TRAIN_SIZE}"
    --val-size "${VAL_SIZE}"
    --seed "${SEED}"
    --max-metric-calls "${MAX_METRIC_CALLS}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --coev-mode "${COEV_MODE}"
    --coev-v2-mode "${COEV_V2_MODE}"
    --adversary-mode "${ADVERSARY_MODE}"
    --eval-method "${EVAL_METHOD}"
    --refusal-threshold "${REFUSAL_THRESHOLD}"
    --asr-threshold "${ASR_THRESHOLD}"
    --gepa-results-dir "${GEPA_RESULTS_DIR}"
    --coev-results-dir "${COEV_RESULTS_DIR}"
    --adversary-results-dir "${ADVERSARY_RESULTS_DIR}"
  )

  if [[ -n "${DEVICE}" ]]; then
    cmd+=(--device "${DEVICE}")
  fi
  if [[ -n "${SAVE_DIR}" ]]; then
    cmd+=(--save-dir "${SAVE_DIR}")
  fi

  "${cmd[@]}"
}

if [[ "${MODE}" != "gepa" && "${MODE}" != "coev_v2" && "${MODE}" != "coev" && "${MODE}" != "adversary" ]]; then
  echo "Unsupported MODE=${MODE}. Expected one of: gepa, coev, coev_v2, adversary"
  exit 1
fi

echo "Starting vLLM (shared reflection + target) for mode=${MODE}..."
cleanup_vllm

uv run python -m vllm.entrypoints.openai.api_server \
  --model "${REFLECTION_MODEL}" \
  --served-model-name "${REFLECTION_MODEL}" \
  --host 0.0.0.0 \
  --port "${REFLECTION_PORT}" \
  --max-model-len "${REFLECTION_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${REFLECTION_GPU_UTIL}" \
  --enforce-eager > logs/unified_reflection_vllm.log 2>&1 &
wait_for_port "${REFLECTION_PORT}" "vLLM"

export REFLECTION_VLLM_BASE_URL="http://127.0.0.1:${REFLECTION_PORT}/v1"
export REFLECTION_VLLM_API_KEY="${REFLECTION_VLLM_API_KEY:-EMPTY}"
echo "REFLECTION_VLLM_BASE_URL=${REFLECTION_VLLM_BASE_URL}"

run_unified

echo "Run complete."
echo "Logs: logs/unified_reflection_vllm.log"
echo "Artifacts: ${GEPA_RESULTS_DIR}, ${COEV_RESULTS_DIR}, ${ADVERSARY_RESULTS_DIR}"
