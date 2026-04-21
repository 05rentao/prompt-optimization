#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:."

# Unified launcher for a single H100-80GB Prime instance.
# Starts one vLLM OpenAI server (REFLECTION_MODEL) used for GEPA reflection and target generation.
# All experiment settings: configs/default.yaml or PROMPT_OPT_CONFIG_PATH (see scripts.unified_runner).
# Shell only sets MODE and vLLM / infra below.
#
# Modes (scripts/run_unified_experiment.py --mode → runs/*.py):
#   - gepa:           GEPA prompt optimization (runs/gepa_run.py)
#   - coev_v2:        CoEV v2 REINFORCE + dual-role GEPA (runs/coev_v2_run.py)
#   - coev_v2_rloo:   CoEV v2 RLOO + dual-role GEPA (same script, --adversary-policy rloo)
#   - adversary:      adversary-only training (runs/adversary_run.py)
#   - xstest:         XSTest over-refusal eval (runs/xstest_run.py)
#   - baseline:       no-adversary HarmBench baseline (runs/baseline_run.py)
#   (runs/coev_run.py legacy is not wired here.)

MODE="${MODE:-gepa}"                         # gepa | coev_v2 | coev_v2_rloo | adversary | xstest | baseline

REFLECTION_PORT="${REFLECTION_PORT:-8001}"

REFLECTION_MODEL="${REFLECTION_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# Conservative split for one H100.
REFLECTION_GPU_UTIL="${REFLECTION_GPU_UTIL:-0.40}"
REFLECTION_MAX_MODEL_LEN="${REFLECTION_MAX_MODEL_LEN:-8192}"

# Optional: path to a Jinja chat template to pass to vLLM via --chat-template.
# Required for base models whose tokenizer_config.json ships without a chat
# template (e.g. lmsys/vicuna-13b-v1.5 — use configs/chat_templates/vicuna_v1.1.jinja).
# Leave empty for models whose tokenizer already defines a template
# (Llama-3.1-Instruct, Llama-2-chat, Qwen2.5-Instruct, etc.).
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

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

if [[ "${MODE}" != "gepa" && "${MODE}" != "coev_v2" && "${MODE}" != "coev_v2_rloo" && "${MODE}" != "adversary" && "${MODE}" != "xstest" && "${MODE}" != "baseline" ]]; then
  echo "Unsupported MODE=${MODE}. Expected one of: gepa, coev_v2, coev_v2_rloo, adversary, xstest, baseline"
  exit 1
fi

echo "Starting vLLM (shared reflection + target) for mode=${MODE}..."
cleanup_vllm

VLLM_EXTRA_ARGS=()
if [[ -n "${CHAT_TEMPLATE}" ]]; then
  if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
    echo "CHAT_TEMPLATE=${CHAT_TEMPLATE} not found; aborting." >&2
    exit 1
  fi
  echo "Using chat template: ${CHAT_TEMPLATE}"
  VLLM_EXTRA_ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi

uv run python -m vllm.entrypoints.openai.api_server \
  --model "${REFLECTION_MODEL}" \
  --served-model-name "${REFLECTION_MODEL}" \
  --host 0.0.0.0 \
  --port "${REFLECTION_PORT}" \
  --max-model-len "${REFLECTION_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${REFLECTION_GPU_UTIL}" \
  "${VLLM_EXTRA_ARGS[@]}" \
  --enforce-eager > logs/unified_reflection_vllm.log 2>&1 &
wait_for_port "${REFLECTION_PORT}" "vLLM"

export REFLECTION_VLLM_BASE_URL="http://127.0.0.1:${REFLECTION_PORT}/v1"
export REFLECTION_VLLM_API_KEY="${REFLECTION_VLLM_API_KEY:-EMPTY}"
echo "REFLECTION_VLLM_BASE_URL=${REFLECTION_VLLM_BASE_URL}"

uv run python scripts/run_unified_experiment.py --mode "${MODE}"

echo "Run complete."
echo "Logs: logs/unified_reflection_vllm.log"
echo "Settings: configs/default.yaml → scripts.unified_runner (or PROMPT_OPT_CONFIG_PATH)."
