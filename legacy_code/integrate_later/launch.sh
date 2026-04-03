#!/bin/bash

# --- CONFIGURATION ---
TARGET_PORT=8000
INSTRUCTOR_PORT=8001
JUDGE_PORT=8002

TARGET_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
# NEW: Llama 3.1 8B. Note: You must be logged into Hugging Face to use this.
INSTRUCTOR_MODEL="unsloth/Llama-3.1-8B-Instruct"
JUDGE_MODEL="cais/HarmBench-Llama-2-13b-cls"

mkdir -p logs

# Ensure your HF Token is available if the model is gated
# export HF_TOKEN="your_token_here"

cleanup() {
    echo "🛑 Cleaning up processes..."
    pkill -f vllm
    fuser -k 8000/tcp 8001/tcp 8002/tcp
}

cleanup

wait_for_port() {
    local port=$1
    echo "⏳ Waiting for port $port to open..."
    while ! nc -z localhost $port; do
        sleep 2
    done
    echo "✅ Port $port is active."
}

echo "🚀 Starting Stable H100 Pipeline..."

# 1. Target Model (0.5B) - Very low footprint
echo "📦 Loading Target..."
uv run python -m vllm.entrypoints.openai.api_server \
    --model $TARGET_MODEL \
    --port $TARGET_PORT \
    --gpu-memory-utilization 0.1 \
    --enforce-eager > logs/target.log 2>&1 &
wait_for_port $TARGET_PORT

# 2. Instructor Model (Llama-3.1-8B) 
echo "🧠 Loading Instructor (Llama 8B)..."
uv run python -m vllm.entrypoints.openai.api_server \
    --model $INSTRUCTOR_MODEL \
    --port $INSTRUCTOR_PORT \
    --gpu-memory-utilization 0.3 \
    --max-model-len 8192 \
    --enforce-eager \
    --served-model-name "Qwen/Qwen2.5-14B-Instruct" > logs/instructor.log 2>&1 &
wait_for_port $INSTRUCTOR_PORT

# 3. Judge Model (13B) - Using quantization to prevent the "-27GB" crash
echo "⚖️ Loading Judge (4-bit Quantized)..."
uv run python -m vllm.entrypoints.openai.api_server \
    --model $JUDGE_MODEL \
    --port $JUDGE_PORT \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.3 \
    --enforce-eager > logs/judge.log 2>&1 &
wait_for_port $JUDGE_PORT

echo "🔥 ALL SYSTEMS GO. Starting DSPy Pipeline..."
uv run python runs/gepa_run.py
