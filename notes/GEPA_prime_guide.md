# 🚀 The Complete H100 Safety Pipeline

> This guide is designed for your **H100 80GB Prime Intellect instance**. It covers everything from the moment you rent the machine to saving your optimized prompt.

---

## Phase 1: Instance Setup & Repository

Check make sure to checkout the H100 80GB x 1 GPU on Prime Intellect

Once your Prime Intellect instance is "Running," connect via SSH.

### 1. SSH into the machine

```bash
ssh root@<YOUR_INSTANCE_IP>
```

### 2. Authenticate with GitHub

Since this is a headless server, the easiest way is using a **Personal Access Token (PAT)**.

- Go to **GitHub Settings → Developer Settings → Tokens (Classic) → Generate Token**
- Grant the `repo` scope

```bash
git clone https://<USERNAME>:<TOKEN>@github.com/05rentao/prompt-optimization.git
git checkout baselines-edit
cd prompt-optimization
```

### 3. Environment Setup

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Sync dependencies
uv sync
mkdir -p logs outputs data

# Set HF token for instructor model
HF_TOKEN=hf_xxx
```

---

## Phase 2: Launching the Triple-Engine

You need to run the `launch.sh` script — the **"Orchestrator"** that carves up your 80GB VRAM.

```bash
chmod +x scripts/launch.sh
./scripts/launch.sh
```

### What this does

| Engine | Model Size | Port | Role |
|--------|-----------|------|------|
| **Student** | 0.5B | 8000 | Fast; handles high-volume jailbreak attempts |
| **Teacher** | 8B | 8001 | Rewrites the system prompt when the student fails |
| **Judge** | 13B | 8002 | Official grader for every response |

---

## Phase 3: Monitoring & Results

While `launch.sh` is running, open a new terminal to monitor hardware and logs.

**Watch VRAM:**
```bash
nvidia-smi -l 1
```
You should see ~55–65GB used.

**Check Teacher's thoughts (GEPA prompt rewrites):**
```bash
tail -f logs/instructor.log
```

**Output:** Once finished, all results will be saved to the `outputs/` directory:

| File / Folder | Description |
|---------------|-------------|
| `optimized_safety_system.json` | The final, production-ready optimized system prompt and signature |
| `detailed_eval_results.csv` | A full breakdown of every test case, including the attack behavior, model response, and safety score |
| `final_score.txt` | A quick-reference summary comparing the Baseline vs. Optimized safety scores |
| `gepa_logs/` | *(Directory)* The "Family Tree" of the evolution process, containing every intermediate prompt variation tested |

---

## 🖥️ Hardware Allocation (H100 80GB)

To maximize the throughput of the evolutionary loop, the 80GB VRAM is partitioned across three distinct services. This **"Triple-Engine"** setup ensures that the Target is fast, the Teacher is smart, and the Judge is consistent.

| Component | Model | Role | VRAM Allocation | Slice (utilization) | KV Cache Room | Port |
|-----------|-------|------|-----------------|---------------------|---------------|------|
| **Target (Student)** | `Qwen2.5-0.5B` |  Student | ~0.93 GiB | 10% (8GB) | ~7.0 GiB | `8000` |
| **Instructor (Teacher)** | `Llama3.1-8B-Instruct` | Teacher | ~16.0 GiB | 30% (24GB) | ~8.0 GiB | `8001` |
| **Judge (Grader)** | `Llama-2-13b-cls` | Grader | ~8.0 GiB (4-bit) | 30% (24GB) | ~16.0 GiB | `8002` |
| **System Overhead** | N/A | OS / CUDA | ~4.0 GiB | 30% (Free) | N/A | N/A

> [!IMPORTANT]
> This distribution is strictly managed via the `launch.sh` script using the `--gpu-memory-utilization` flag in vLLM. This prevents VRAM fragmentation and Out of Memory (OOM) errors during the highly concurrent `dspy.GEPA` optimization phase.

### 💡 Why this setup?

**0.5B Student:** Optimized for low-latency inference, allowing the pipeline to test hundreds of prompts per minute.

**14B Teacher:** Provides the "meta-cognitive" reasoning required to look at a failed refusal and generate a more robust defense strategy.

**13B Judge:** The industry-standard HarmBench classifier ensures that our optimization metric is grounded in peer-reviewed safety standards.

### One Last Step Before You Run

Since you're about to fire this up on the H100, make sure you check your `nvidia-smi` right after running `./launch.sh`. If you see usage sitting around **72/80GB**, you are in the perfect "Goldilocks" zone. ✅

---

## 🐛 Common Bugs & Troubleshooting

| Bug / Error | Cause | Fix |
|-------------|-------|-----|
| **Connection Refused (800x)** | vLLM server hasn't finished loading weights | The `launch.sh` wait-loop should handle this; if it persists, check `logs/judge.log` for OOM during startup |
| **CUDA Out of Memory** | Sum of `--gpu-memory-utilization` exceeded 1.0 (100%) | Ensure three servers are set to `0.1`, `0.4`, and `0.4` — leaving ~10% (~8GB) for OS and Python overhead |
| **Permission Denied** | `launch.sh` or `clean.sh` aren't executable | Run `chmod +x <filename>` |
| **Zombies on Port 8000** | Previous run crashed without killing servers | Run `pkill -f vllm` or `fuser -k 8000/tcp 8001/tcp 8002/tcp` |
| **Empty CSV / 0-byte data** | HarmBench download failed or directory missing | Run `mkdir -p data` before the loader; check with `ls -lh data/` |

If the script hangs or crashes, your GPU memory might still be "zombie-locked" by the failed processes. Run these in order to reset:

A. The "Kill Everything" Command:
Sometimes pkill isn't enough. This finds exactly what is touching your GPU and terminates it:

```bash
fuser -v /dev/nvidia* # See who is using the GPU
sudo fuser -k /dev/nvidia* # Force kill them
```

B. Check the Logs in Real-Time:
Instead of waiting for the script to finish, "tail" the logs to see the memory math as it happens:
```bash
tail -f logs/instructor.log logs/judge.log
```

---

## 🛠️ Maintenance Tips

### Save Your Results Before Terminating

Before you terminate the Prime Intellect instance, download your `outputs/` and `logs/` folders using `scp` or push them to a "results" branch on GitHub. **Once the instance is deleted, the data is gone forever.**

```bash
# Example: download results to your local machine
scp -r root@<YOUR_INSTANCE_IP>:~/prompt-optimization/outputs ./
scp -r root@<YOUR_INSTANCE_IP>:~/prompt-optimization/logs ./
```

### ASR Calculation

If your **ASR (Attack Success Rate)** doesn't drop, check whether your `safety_metric` is correctly inverting the reward.

> **Remember:** Success for you = Judge says **"No"** to the jailbreak attempt.
