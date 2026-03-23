#!/usr/bin/env bash
set -euo pipefail

# Prime launcher for runs/coev_v2_RLOO_run.py
#
# What this script does:
# 1) Ensures uv environment is available
# 2) Starts vLLM: GEPA reflection and target generation share this OpenAI server
# 3) Runs coev_v2 RLOO pipeline (hyperparameters from configs/default.yaml or PROMPT_OPT_CONFIG_PATH)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# 8001 often conflicts on RunPod / shared hosts (proxy or stale bind). Override with REFLECTION_PORT if needed.
REFLECTION_PORT="${REFLECTION_PORT:-8765}"
REFLECTION_HTTP_WAIT_S="${REFLECTION_HTTP_WAIT_S:-900}"
REFLECTION_MODEL="${REFLECTION_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# Single-GPU: adversary + HarmBench judge (4-bit NF4 by default) + vLLM (reflection + target HTTP, no local target).
# Tune if vLLM OOMs or is slow; local target weights are no longer loaded in the Python process.
REFLECTION_GPU_UTIL="${REFLECTION_GPU_UTIL:-0.50}"
REFLECTION_MAX_MODEL_LEN="${REFLECTION_MAX_MODEL_LEN:-8192}"

DEVICE="${DEVICE:-}"
SAVE_DIR="${SAVE_DIR:-}"

INITIAL_ATTACKER_INSTRUCTION="${INITIAL_ATTACKER_INSTRUCTION:-}"
INITIAL_DEFENSE_PROMPT="${INITIAL_DEFENSE_PROMPT:-}"

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

# True when something accepts TCP on 127.0.0.1:port (uses nc if present, else bash /dev/tcp).
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
  local timeout_s="${3:-240}"
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

# Waits for TCP on port; fails fast if vLLM exits (e.g. HF 401 gated model) instead of waiting full timeout.
wait_for_vllm_reflection_port() {
  local port="$1"
  local name="$2"
  local timeout_s="${3:-240}"
  local waited=0
  local vllm_log="${ROOT_DIR}/logs/coev_v2_rloo_reflection_vllm.log"
  echo "Waiting for ${name} on :${port} (timeout ${timeout_s}s)..."
  until port_is_open "${port}"; do
    sleep 2
    waited=$((waited + 2))
    if [[ -f "${vllm_log}" ]] && grep -qE "GatedRepoError|huggingface_hub\.errors\.GatedRepoError|Cannot access gated repo|You are trying to access a gated repo" "${vllm_log}" 2>/dev/null; then
      echo "ERROR: Hugging Face rejected access to the reflection model (gated repo or invalid/missing token)."
      echo "Fix: export HF_TOKEN or HUGGINGFACE_HUB_TOKEN, run 'huggingface-cli login', and accept the model license on Hugging Face."
      echo "Model: ${REFLECTION_MODEL}"
      echo "--- tail ${vllm_log} ---"
      tail -n 30 "${vllm_log}"
      return 1
    fi
    if [[ -n "${REFLECTION_VLLM_PID:-}" ]] && ! kill -0 "${REFLECTION_VLLM_PID}" 2>/dev/null; then
      echo "ERROR: ${name} process exited before binding :${port} (see log below)."
      if [[ -f "${vllm_log}" ]]; then
        echo "--- tail ${vllm_log} ---"
        tail -n 40 "${vllm_log}"
      fi
      return 1
    fi
    if [[ "${waited}" -ge "${timeout_s}" ]]; then
      echo "ERROR: ${name} did not become ready on :${port} in time."
      echo "If the model is large, ensure weights are cached; if you see HF errors above, fix token/license."
      return 1
    fi
  done
  echo "${name} is up on :${port}"
}

# TCP can be open before vLLM serves OpenAI JSON; hosted envs may return HTML (e.g. proxy 502) until the backend is ready.
# Use project venv Python (not "curl | uv run python"): piping into uv can be flaky; large models may need many minutes.
wait_for_openai_models_json() {
  local port="$1"
  local name="$2"
  local timeout_s="${3:-900}"
  local waited=0
  local py="${ROOT_DIR}/.venv/bin/python"
  if [[ ! -x "${py}" ]]; then
    py="uv run python"
  fi
  echo "Waiting for ${name} OpenAI GET /v1/models JSON on :${port} (timeout ${timeout_s}s)..."
  echo "(Progress updates every ~3s; vLLM log snippet every 15s — see logs/coev_v2_rloo_reflection_vllm.log for full output.)"
  while [[ "${waited}" -lt "${timeout_s}" ]]; do
    local body
    body="$(curl -sS -m 30 -H "Authorization: Bearer EMPTY" "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)"
    if echo "${body}" | "${py}" -c "
import json, sys
raw = sys.stdin.read().strip()
if not raw or raw.lstrip().startswith('<'):
    sys.exit(1)
d = json.loads(raw)
if not isinstance(d.get('data'), list):
    sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
      echo "${name} OpenAI /v1/models is ready on :${port} (after ${waited}s)"
      return 0
    fi
    local hint
    hint="$(echo "${body}" | "${py}" -c "
import json, sys
raw = sys.stdin.read()
s = raw.strip()
if not s:
    print('no HTTP body yet (connection timeout, or server not speaking HTTP)')
elif s.lstrip().startswith('<'):
    print('got HTML, not JSON (proxy/502 page or wrong service on port)')
else:
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        print('body not JSON:', str(e)[:100])
    else:
        dd = d.get('data')
        if not isinstance(dd, list):
            print('JSON has no data[] list; keys:', list(d.keys())[:8])
        else:
            print('valid JSON (unexpected in fail path)')
" 2>/dev/null || echo 'could not classify response')"
    printf '  [%3ss / %ss] %s\n' "${waited}" "${timeout_s}" "${hint}"
    if (( waited % 15 == 0 && waited > 0 )) && [[ -f logs/coev_v2_rloo_reflection_vllm.log ]]; then
      local logline
      logline="$(tail -n 1 logs/coev_v2_rloo_reflection_vllm.log 2>/dev/null | tr -d '\r' | cut -c1-160)"
      if [[ -n "${logline}" ]]; then
        printf '  vLLM log (last line): %s\n' "${logline}"
      fi
    fi
    sleep 3
    waited=$((waited + 3))
  done
  echo "ERROR: ${name} did not return valid OpenAI JSON from /v1/models on :${port} in time."
  echo "Last response from http://127.0.0.1:${port}/v1/models (first 500 chars):"
  body="$(curl -sS -m 30 -H "Authorization: Bearer EMPTY" "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)"
  if [[ -n "${body}" ]]; then
    echo "${body}" | head -c 500
    echo ""
  else
    echo "(empty — connection failed or timed out)"
  fi
  if [[ -f logs/coev_v2_rloo_reflection_vllm.log ]]; then
    echo "--- tail logs/coev_v2_rloo_reflection_vllm.log ---"
    tail -n 40 logs/coev_v2_rloo_reflection_vllm.log
  fi
  echo "If you see HTML/502 (e.g. RunPod), fix proxy/port. If vLLM is still loading weights, increase wait (3rd arg) or set REFLECTION_HTTP_WAIT_S."
  return 1
}

cleanup() {
  if [[ "${KEEP_VLLM_UP}" == "1" ]]; then
    echo "KEEP_VLLM_UP=1 set; leaving vLLM server running."
    return
  fi
  echo "Cleaning up reflection vLLM process..."
  if [[ -n "${REFLECTION_VLLM_PID:-}" ]]; then
    kill "${REFLECTION_VLLM_PID}" 2>/dev/null || true
  fi
  fuser -k "${REFLECTION_PORT}/tcp" >/dev/null 2>&1 || true
}
# trap cleanup EXIT
cleanup
echo "--------------------------------"

echo "Ensuring reflection port is free..."
fuser -k "${REFLECTION_PORT}/tcp" >/dev/null 2>&1 || true
sleep 2

echo "Starting reflection vLLM (${REFLECTION_MODEL}) on :${REFLECTION_PORT}..."
: > logs/coev_v2_rloo_reflection_vllm.log
uv run python -m vllm.entrypoints.openai.api_server \
  --model "${REFLECTION_MODEL}" \
  --served-model-name "${REFLECTION_MODEL}" \
  --host 0.0.0.0 \
  --port "${REFLECTION_PORT}" \
  --max-model-len "${REFLECTION_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${REFLECTION_GPU_UTIL}" \
  --enforce-eager >> logs/coev_v2_rloo_reflection_vllm.log 2>&1 &
REFLECTION_VLLM_PID=$!
# Fail fast if vLLM cannot bind (port in use — e.g. RunPod proxy on 8001).
sleep 4
if grep -q "Address already in use" logs/coev_v2_rloo_reflection_vllm.log 2>/dev/null; then
  echo "ERROR: vLLM could not bind :${REFLECTION_PORT} (address already in use)."
  echo "Pick a free port:  REFLECTION_PORT=8777 ./scripts/launch_coev_v2_rloo_prime.sh"
  exit 1
fi
wait_for_vllm_reflection_port "${REFLECTION_PORT}" "reflection vLLM" 240
wait_for_openai_models_json "${REFLECTION_PORT}" "reflection vLLM" "${REFLECTION_HTTP_WAIT_S}"

export REFLECTION_VLLM_BASE_URL="http://127.0.0.1:${REFLECTION_PORT}/v1"
export REFLECTION_VLLM_API_KEY="${REFLECTION_VLLM_API_KEY:-EMPTY}"
echo "Reflection OpenAI base URL (for coev run): ${REFLECTION_VLLM_BASE_URL}"

# Reduces fragmentation OOMs when many models share one GPU (PyTorch 2.x).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Launching runs/coev_v2_RLOO_run.py..."
RUN_CMD=(uv run python runs/coev_v2_RLOO_run.py)

if [[ -n "${MODE:-}" ]]; then
  RUN_CMD+=(--mode "${MODE}")
fi
if [[ -n "${RESULTS_DIR:-}" ]]; then
  RUN_CMD+=(--results-dir "${RESULTS_DIR}")
fi
if [[ -n "${TRAINING_CSV_NAME:-}" ]]; then
  RUN_CMD+=(--training-csv-name "${TRAINING_CSV_NAME}")
fi
if [[ -n "${DEVICE}" ]]; then
  RUN_CMD+=(--device "${DEVICE}")
fi
if [[ -n "${SAVE_DIR}" ]]; then
  RUN_CMD+=(--save-dir "${SAVE_DIR}")
fi
if [[ -n "${INITIAL_ATTACKER_INSTRUCTION}" ]]; then
  RUN_CMD+=(--initial-attacker-instruction "${INITIAL_ATTACKER_INSTRUCTION}")
fi
if [[ -n "${INITIAL_DEFENSE_PROMPT}" ]]; then
  RUN_CMD+=(--initial-defense-prompt "${INITIAL_DEFENSE_PROMPT}")
fi

"${RUN_CMD[@]}"

echo "Run complete."
echo "Reflection vLLM log: logs/coev_v2_rloo_reflection_vllm.log"
echo "Artifacts: see runs.coev_v2 in configs/default.yaml (or your PROMPT_OPT_CONFIG_PATH)."
