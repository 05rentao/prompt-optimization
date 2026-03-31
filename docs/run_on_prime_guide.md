# 🚀 The Complete H100 Safety Pipeline

> This guide is designed for your **H100 80GB Prime Intellect instance**. It covers everything from the moment you rent the machine to saving your optimized prompt.

---

## Phase 1: Instance Setup & Repository

Check make sure to checkout the H100 80GB x 1 GPU on Prime Intellect

Once your Prime Intellect instance is "Running," connect via SSH.

### 1. Create and SSH prime pod


```bash
prime availability list --gpu-type H100_80GB
```

Pick one you like and copy its ID, and initialize the pod. Note: if it says "spot" this means that the GPU can spontaniously deactivate, pick a non-spot instance for stable performance. Feel free to also change the `--name` argument.

```bash
prime pods create --id <ID> --disk-size 500 --name run-gepa 
```

Once created, it will output a pod id of the instance you just created. To check the status of your pod and whether it's active, use:
```bash
prime pods status <Pod_ID>
```

Once the 'Status' says `ACTIVE` and `Installation Status` says ``FINISHED`, you can SSH into the machine. Copy the `SSH` string in the status output, and use the following command.

```bash
ssh <SSH_string>
```

### 2. Authenticate with GitHub

Since this is a headless server, the easiest way is using a **Personal Access Token (PAT)**.

- Go to **GitHub Settings → Developer Settings → Tokens (Classic) → Generate Token**
- Grant the `repo` scope

```bash
git clone https://<USERNAME>:<TOKEN>@github.com/05rentao/prompt-optimization.git
cd prompt-optimization
git checkout <whatever branch you are working on>
```

### 3. Environment Setup

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Sync dependencies
uv sync

# Set HF token for model initialization
export HF_TOKEN=hf_xxx
```

### 3.5 Make sure you have access to models/datasets on Hugging Face

Before launching long runs, verify your token and access:

1. Open each required model/dataset page on Hugging Face and click **Accept** if gated.
2. Run the checks below on the Prime instance:

```bash
python3 -c "from huggingface_hub import whoami; print(whoami())"

python3 - <<'PY'
from huggingface_hub import model_info, dataset_info

models = [
    "unsloth/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "cais/HarmBench-Mistral-7b-val-cls",
]
datasets = [
    "walledai/HarmBench",
]

for m in models:
    try:
        model_info(m)
        print(f"MODEL OK: {m}")
    except Exception as e:
        print(f"MODEL FAIL: {m} -> {e}")

for d in datasets:
    try:
        dataset_info(d)
        print(f"DATASET OK: {d}")
    except Exception as e:
        print(f"DATASET FAIL: {d} -> {e}")
PY
```

### 4 Connect Cursor/VS Code to Prime (after CLI login works)

#### 4.1 Add SSH host entry on your local machine

```bash
nano ~/.ssh/config
```

Paste:

```text
Host prime-pod
    HostName <SSH_IP_ADDRESS>
    User <the string before the @ symbol in the ssh>
    Port <differnt port number if applicable, usually 22>
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 120
```

#### 4.2 Create a key if needed, then copy your public key (local machine)

```bash
# (optional) create key
test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -C "prime-access"

# print key to terminal, copy this!
cat ~/.ssh/id_ed25519.pub
```

#### 4.3 Add your public key to the Prime instance

In the terminal already connected to Prime:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "<PASTE_YOUR_PUBLIC_KEY_STRING_HERE>" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```
Test from your local machine:

```bash
ssh prime-pod
```

#### 4.4 Connect from Cursor/VS Code

1. Install the **Remote - SSH** extension.
2. Open command palette (`Ctrl/Cmd + Shift + P`).
3. Run `Remote-SSH: Connect to Host...`.
4. Select `prime-pod`.
5. Open `~/prompt-optimization`.

### 5 Terminate the Prime pod safely

You can terminate from the Prime dashboard, or by CLI:

```bash
prime pods status <Pod_ID>

# delete the prime dashboard.
prime pods terminate <Pod_ID>
```

### Notes:

- If you make code changes on Prime, commit and push to git.
- Before deleting the pod, download or push `results/`, `outputs/`, and `logs/`.
- If you only disconnect SSH and keep the pod alive, files usually persist. If the pod is deleted or preempted, local pod storage is gone.

---

## Phase 2: Launch runs with current scripts

This repo now uses the unified launcher and run scripts in `scripts/` and `runs/`.

### 6) Recommended launch commands

```bash
# Main pipeline (CoEV v2)
MODE=coev_v2 bash scripts/launch_unified_prime.sh

# GEPA-only
MODE=gepa bash scripts/launch_unified_prime.sh

# CoEV legacy
MODE=coev bash scripts/launch_unified_prime.sh

# Adversary-only
MODE=adversary bash scripts/launch_unified_prime.sh
```

Edit `configs/default.yaml` (or `PROMPT_OPT_CONFIG_PATH`) for sub-modes, budgets, and paths — see `scripts.unified_runner` and `runs.*`.

### 7) Typical runtime expectations on one H100 (rough guidance)

These are ballpark estimates and vary by model load time, dataset size, and `runs.gepa.max_metric_calls` / CoEV settings in YAML:

- `MODE=adversary` (default train sizes in config): often around 20-60 minutes.
- `MODE=coev`: often around 30-90 minutes.
- `MODE=gepa`: often around 1-3+ hours.
- `MODE=coev_v2` (GEPA stages enabled): often around 1.5-4+ hours.

If you want quick validation first, use smoke mode:

```bash
bash scripts/launch_smoke_prime.sh
```

---

## Phase 3: Monitoring during runs

### 8) Watch process and GPU health

```bash
# GPU utilization and VRAM
nvidia-smi -l 2

# See active python/vLLM processes
ps -ef | rg "vllm|run_unified_experiment|coev_v2_run|gepa_run|adversary_run"
```

### 9) Follow logs in real time

The unified Prime launcher writes reflection server logs to:

- `logs/unified_reflection_vllm.log` (for `MODE=gepa` and `MODE=coev_v2`)

```bash
tail -f logs/unified_reflection_vllm.log
```

Useful pattern checks:

```bash
# reflection server startup / errors
rg "error|ERROR|Traceback|CUDA|OOM|started|Uvicorn" logs/unified_reflection_vllm.log
```

### 10) Check intermediate outputs while job is running

```bash
# list fresh artifacts as they appear
ls -lt results/

# for coev_v2 runs
ls -lt results/coev | head

# for gepa runs
ls -lt results/gepa | head
```

---

## Phase 4: Results and artifact interpretation

### 11) Primary artifact locations

- GEPA mode: `results/gepa/`
- CoEV mode: `results/coev/`
- CoEV v2 mode: `results/coev/` (or your `COEV_RESULTS_DIR`)
- Adversary mode: `results/adversary/`
- Smoke runs: `results/smoke/<RUN_TAG>/...`

### 12) What to inspect first

1. `run_manifest.json` for reproducibility and run settings.
2. metrics JSON file for baseline vs optimized ASR/refusal summaries.
3. CSV outputs for row-level debugging (`prompt`, rewritten prompt, response, reward/verdict).
4. For GEPA and CoEV v2, trajectory/trace CSV and plots to verify optimization direction.

### 13) Fast sanity checks after completion

```bash
# find run manifests quickly
rg --files -g "**/run_manifest.json" results

# inspect summary metrics files
rg --files -g "**/*metrics*.json" results
```

---

## Common bugs and troubleshooting (current scripts)

| Symptom | Likely cause | Fix |
|---|---|---|
| `Unsupported MODE=...` | Invalid `MODE` env var passed to launcher | Use one of `gepa`, `coev`, `coev_v2`, `adversary` |
| Run hangs at reflection startup | Port conflict or vLLM failed to initialize | Check `logs/unified_reflection_vllm.log`, then kill stale processes and rerun |
| `HF` / dataset access errors | Token missing or model terms not accepted | Export `HF_TOKEN`, run access checks in section 3.5 |
| CUDA OOM during reflection mode | Reflection model footprint too high for current settings | Reduce `REFLECTION_GPU_UTIL` (example `0.30`) or reduce model/context length |
| CoEV v2/GEPA progress is very slow | Large `MAX_METRIC_CALLS`, long generations, or heavy reflection model | Lower `MAX_METRIC_CALLS`, `MAX_TOKENS`, and/or use smoke launch first |
| Permission denied on scripts | Missing execute bit | `chmod +x scripts/launch_unified_prime.sh scripts/launch_smoke_prime.sh` |
| No artifacts where expected | Custom results dirs overridden in env vars | Echo env vars (`GEPA_RESULTS_DIR`, `COEV_RESULTS_DIR`, `ADVERSARY_RESULTS_DIR`) before launch |

### 14) Useful recovery commands

```bash
# Stop reflection vLLM and free port
pkill -f "vllm.entrypoints.openai.api_server" || true
fuser -k 8001/tcp || true

# If GPU is still locked by stale processes
fuser -v /dev/nvidia*
sudo fuser -k /dev/nvidia* || true
```

---

## Maintenance tips for Prime runs

### 15) Keep runs reproducible

- Record launch command and env overrides in a text file per run.
- Keep `run_manifest.json` with your results when sharing outcomes.
- Prefer branch-per-experiment and commit config/CLI changes early.

### 16) Preserve artifacts before deleting pod

```bash
# Example: copy artifacts from Prime to local machine
scp -r root@<YOUR_INSTANCE_IP>:~/prompt-optimization/results ./results_prime_backup
scp -r root@<YOUR_INSTANCE_IP>:~/prompt-optimization/logs ./logs_prime_backup
```

### 17) Control disk usage on long experiments

```bash
du -sh results logs
df -h
```

Clean large stale outputs when safe:

```bash
rm -rf results/smoke/*
```