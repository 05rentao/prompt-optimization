#!/bin/bash

set -euo pipefail

# --- CONFIGURATION ---

export PYTHONPATH="${PYTHONPATH:-}:."

JUDGE_PORT=8002
JUDGE_MODEL="cais/HarmBench-Llama-2-13b-cls"
JUDGE_GPU_UTIL=0.45

STEERING_SCRIPT="scripts/steer_results.py"
LOG_DIR="logs"
JUDGE_LOG="$LOG_DIR/judge.log"
STEERING_LOG="$LOG_DIR/steer_results.log"

# If your model access is gated, export this before running:
# export HF_TOKEN="hf_xxx"

mkdir -p "$LOG_DIR" outputs vectors data

cleanup() {
    echo "Cleaning up background vLLM services..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    fuser -k ${JUDGE_PORT}/tcp || true
}

wait_for_port() {
    local port="$1"
    echo "Waiting for localhost:${port}..."
    until nc -z localhost "$port"; do
        sleep 2
    done
    echo "Port ${port} is active."
}

trap cleanup EXIT

echo "Resetting existing services on port ${JUDGE_PORT}..."
cleanup

echo "Starting HarmBench judge on port ${JUDGE_PORT}..."
uv run python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --port "$JUDGE_PORT" \
    --gpu-memory-utilization "$JUDGE_GPU_UTIL" \
    --enforce-eager > "$JUDGE_LOG" 2>&1 &

wait_for_port "$JUDGE_PORT"

echo "Running activation steering evaluation..."
uv run python "$STEERING_SCRIPT" "$@" | tee "$STEERING_LOG"

echo "Done."
echo "Steering log: $STEERING_LOG"
echo "Judge log: $JUDGE_LOG"
