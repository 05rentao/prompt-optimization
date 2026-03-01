#!/bin/bash

# --- CONFIGURATION ---
# Ports for the three different services
TARGET_PORT=8000
INSTRUCTOR_PORT=8001
JUDGE_PORT=8002

# Model Names
TARGET_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTOR_MODEL="Qwen/Qwen2.5-14B-Instruct"
JUDGE_MODEL="cais/HarmBench-Llama-2-13b-cls"

# --- HELPER FUNCTIONS ---
wait_for_server() {
    local port=$1
    local name=$2
    echo "⏳ Waiting for $name to be ready on port $port..."
    while ! curl -s http://localhost:$port/v1/models > /dev/null; do
        sleep 5
    done
    echo "✅ $name is LIVE."
}

cleanup() {
    echo "🛑 Shutting down all vLLM servers..."
    pkill -f vllm
    exit
}

# Ensure cleanup happens if you Ctrl+C the script
trap cleanup SIGINT SIGTERM

# --- START SERVERS ---
echo "🚀 Initializing H100 Pipeline..."

# 1. Target Model (Student) - Low VRAM
python -m vllm.entrypoints.openai.api_server \
    --model $TARGET_MODEL \
    --port $TARGET_PORT \
    --gpu-memory-utilization 0.1 > logs/target.log 2>&1 &
echo "📦 Started Target Model (PID: $!)"

# 2. Instructor Model (Teacher) - Medium VRAM
python -m vllm.entrypoints.openai.api_server \
    --model $INSTRUCTOR_MODEL \
    --port $INSTRUCTOR_PORT \
    --gpu-memory-utilization 0.4 > logs/instructor.log 2>&1 &
echo "🧠 Started Instructor Model (PID: $!)"

# 3. Judge Model (Grader) - Medium VRAM
python -m vllm.entrypoints.openai.api_server \
    --model $JUDGE_MODEL \
    --port $JUDGE_PORT \
    --gpu-memory-utilization 0.4 > logs/judge.log 2>&1 &
echo "⚖️ Started Judge Model (PID: $!)"

# --- VALIDATION ---
mkdir -p logs
wait_for_server $TARGET_PORT "Target"
wait_for_server $INSTRUCTOR_PORT "Instructor"
wait_for_server $JUDGE_PORT "Judge"

echo "🔥 ALL SYSTEMS GO. Starting DSPy Pipeline..."

# --- RUN MAIN SCRIPT ---
uv run scripts/gepa_run.py

# Auto-cleanup after the script finishes
cleanup