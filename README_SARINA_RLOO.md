# sarina-rloo-steerbaseline

Week 10 improvements to adversarial prompt optimization with RLOO, rejection sampling, better adversary prompts, multi-target queries, and XSTest integration.

## What's New

### Core Algorithmic Improvements

**RLOO (Leave-One-Out Reinforcement Learning)**
- Implements variance reduction for multi-sample policy gradients
- For K > 1 samples per prompt, uses leave-one-out baseline: `advantage_i = reward_i - mean(reward_{j != i})`
- Reduces variance vs vanilla REINFORCE without introducing bias
- Flag: `--rloo-n K` (default 1 = vanilla REINFORCE, 5+ recommended)

**Rejection Sampling (Paper §5)**
- Only trains on successful attacks (reward > 0.5)
- Skips gradient update if no successes found (prevents training on zero-signal)
- Optimizes log-probability objective → stronger emphasis on high success rates
- Flags: `--rs-budget M` (max samples per step), `--rs-min-successes N` (success threshold)
- Set `--rs-min-successes 0` to use RLOO mode instead

**Multi-Query Rewards**
- Query target model N times per attack
- Reward = fraction of compliant responses (soft reward in [0, 1/N, 2/N, ..., 1])
- Reduces noise vs binary single-query reward
- Flag: `--target-queries N` (default 1)

### Better Adversary Prompts

**5 Jailbreak Strategies** (selectable via `--adversary-prompt`):
1. **persona** — Professional identity wrap ("I am a forensic chemist...")
2. **decompose** — Subproblem decomposition (2–4 numbered sub-questions)
3. **fictional** — Story/script embedding with technical detail requirement
4. **academic** — Graduate-level research/literature framing
5. **default** — All 5 strategies with detailed examples (recommended)

**Evolved Prompts** (GEPA mode):
- After each stage, meta-generates new attacker/defender system prompts
- Attacker instruction evolves to exploit discovered weaknesses
- Defense prompt hardens against successful attacks
- Leads to arms-race dynamics in safety research

### XSTest Integration

**XSTest Benchmark** (overrefusal detection):
- 250 safe + 250 unsafe prompts from Paul/XSTest (HuggingFace)
- Measures:
  - **Safe Compliance**: Fraction of safe prompts answered (vs overrefusal)
  - **Unsafe ASR**: Fraction of unsafe prompts answered (attack success)
  - **Overrefusal Rate**: Safe prompts incorrectly refused
  - **Underrefusal Rate**: Unsafe prompts incorrectly answered
- Shows if improvements come from evasion or *better prompt crafting*

### Universal Benchmark Runner

`scripts/run_benchmark.py` supports:
- XSTest, HarmBench, or both
- Configurable limits, backends (vllm, hf, openai, litellm)
- Results saved with detailed metrics and raw output

---

## Quick Start on Prime Intellect

### 1. Clone and Checkout

```bash
cd /root
git clone https://github.com/05rentao/prompt-optimization.git
cd prompt-optimization
git checkout sarina-rloo-steerbaseline
```

### 2. Create Environments

**Terminal 1: vLLM server**
```bash
conda create -n vllm-serve python=3.11 -y
conda activate vllm-serve
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.35 \
  --enforce-eager \
  --max-model-len 4096
```

Wait for: `INFO:     Application startup complete.`

**Terminal 2: coev training**
```bash
conda create -n coev-main python=3.11 -y
conda activate coev-main
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install unsloth pandas datasets openai pydantic pyyaml transformers
```

### 3. Run GEPA Training

```bash
cd ~/prompt-optimization
export PYTHONPATH=~/prompt-optimization
export REFLECTION_VLLM_BASE_URL="http://localhost:8000/v1"
export REFLECTION_VLLM_API_KEY="fake-key"

python runs/coev_run.py --mode gepa --adversary-prompt default
```

**Outputs:**
- `smoke_test_gepa.csv` — per-iteration training logs
- Console output — baseline ASR, stage-wise ASR progression

### 4. Run Benchmarks (Optional)

```bash
# XSTest (overrefusal measurement)
uv run python scripts/run_benchmark.py \
  --benchmark xstest \
  --xstest-limit 100 \
  --output-dir results/xstest_latest

# HarmBench (standard safety benchmark)
uv run python scripts/run_benchmark.py \
  --benchmark harmbench \
  --harmbench-limit 100 \
  --output-dir results/harmbench_latest
```

### 5. Save Results

```bash
git add smoke_test_gepa.csv results/
git commit -m "Week 10 results: GEPA with RLOO + XSTest/HarmBench"
git push origin sarina-rloo-steerbaseline
```

## Quick start on prime intellect **uv version**

### 1. Clone and Checkout

```bash
git clone https://github.com/05rentao/prompt-optimization.git
# git clone https://<USERNAME>:<TOKEN>@github.com/05rentao/prompt-optimization.git
# use below command to also authenticate user.

cd prompt-optimization
git checkout sarina-rloo-steerbaseline
```

### 2. Set up venv for dependencies

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Sync dependencies
uv sync

# Set HF token for model initialization
export HF_TOKEN=hf_xxx

mkdir logs/

uv run python -m vllm.entrypoints.openai.api_server \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --served-model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --port 8000 \
  --gpu-memory-utilization 0.35 \
  --enforce-eager \
  --max-model-len 4096 > logs/unified_reflection_vllm.log 2>&1 &


export PYTHONPATH=~/prompt-optimization
export REFLECTION_VLLM_BASE_URL="http://localhost:8000/v1"
export REFLECTION_VLLM_API_KEY="fake-key"
```

Wait for the log line `INFO:     Application startup complete.` 

use `tail -f logs/unified_reflection_vllm.log` to watch live updates of model set up at `http://localhost:8000/v1`

### 3.1. Run GEPA Training

```bash
uv run python runs/coev_run.py --mode gepa --adversary-prompt default
```

**Outputs:**
- `smoke_test_gepa.csv` — per-iteration training logs
- Console output — baseline ASR, stage-wise ASR progression


### 3.2. Run reinforce

```bash
uv run python runs/coev_run.py --mode reinforce --adversary-prompt default --rloo-n 5 --target-queries 2
```

### 4. Run Benchmarks (Optional)

```bash
# XSTest (overrefusal measurement)
uv run python scripts/run_benchmark.py \
  --benchmark xstest \
  --xstest-limit 100 \
  --output-dir results/xstest_latest

# HarmBench (standard safety benchmark)
uv run python scripts/run_benchmark.py \
  --benchmark harmbench \
  --harmbench-limit 100 \
  --output-dir results/harmbench_latest
```

### 5. Save Results

use `git add` and `git commit` and `git push`

---

### see section 4 of `docs/run_on_prime_guide.md`. 
## Configuration

### GEPA Training Config (runs/coev_run.py)

```python
GepaConfig(
    stages: int = 2                      # Evolutionary stages
    iters_per_stage: int = 5             # Training iterations per stage
    eval_every_stages: int = 1           # Eval after every N stages
    train_slice_end: int = 50            # Train on first 50 prompts
    eval_slice_start: int = 100          # Eval on prompts 100–110
    eval_slice_end: int = 110
    rloo_n: int = 1                      # K samples per step (1=vanilla, 5+=RLOO)
    target_queries: int = 1              # Times to query target per attack
    rs_budget: int = 5                   # Max samples per rejection sampling step
    rs_min_successes: int = 1            # Min successes before stopping RS
)
```

### REINFORCE Config (similar structure for single-stage training)

```bash
uv run python runs/coev_run.py --mode reinforce \
  --adversary-prompt persona \
  --rloo-n 5 \
  --target-queries 2 \
  --rs-min-successes 3
```

---

## Expected Results

### GEPA Training (2 stages × 5 iters)
- **Baseline ASR**: 10–20% (before training)
- **Stage 0 ASR**: 20–35% (after first stage)
- **Stage 1 ASR**: 30–50% (after second stage)
- Training time: ~15–20 min total

### XSTest Benchmark (100 safe + 100 unsafe)
```json
{
  "safe_compliance": 0.85–0.95,     // Good if > 0.85 (low overrefusal)
  "unsafe_asr": 0.15–0.35,          // Typical ASR range
  "overrefusal_rate": 0.05–0.15,    // Ideally < 0.10
  "underrefusal_rate": 0.65–0.85    // Typical (high = bad)
}
```

### HarmBench Benchmark
```json
{
  "asr": 0.20–0.40,                 // Typical range
  "num_examples": 100
}
```

**Interpretation:**
- Compare **Baseline ASR** vs **Final ASR** to measure training improvement
- **Safe Compliance** + **Overrefusal Rate** show if model is refusing safe queries
- **Unsafe ASR** shows attack success rate (lower = safer)
- XSTest better at detecting overrefusal than HarmBench

---

## Key Files

- `runs/coev_run.py` — GEPA/REINFORCE training with RLOO, RS, multi-query
- `scripts/run_benchmark.py` — Universal XSTest/HarmBench runner
- `src/dataset/xstest_loader.py` — XSTest loader from HuggingFace
- `src/evaluators.py` — XSTest classification + metrics
- `src/types.py` — XSTestExampleRow type definition

## Differences from Main

| Feature | Main | sarina-rloo-steerbaseline |
|---------|------|---------------------------|
| Adversary prompts | Single prompt | 5 strategies (default, persona, decompose, academic, fictional) |
| Policy gradient | Vanilla REINFORCE | RLOO with leave-one-out baseline |
| Training logic | All samples | Rejection sampling (only successful attacks) |
| Reward signal | Binary (0/1) | Soft (0 to 1, multi-query) |
| Benchmarking | HarmBench only | HarmBench + XSTest (overrefusal detection) |
| Evolved prompts | No | Yes (GEPA mode) |

---

## Troubleshooting

**vLLM connection refused:**
- Check vLLM is running: `curl http://localhost:8000/v1/models`
- Verify env vars set: `echo $REFLECTION_VLLM_BASE_URL`

**torch/transformers version conflict:**
- Use separate conda environments (vllm-serve ≠ coev-main)
- Don't mix pip installations in same env

**Out of memory:**
- Reduce `--gpu-memory-utilization` (default 0.35)
- Reduce `--rloo-n` (fewer samples per step)
- Reduce `max_seq_length` in adversary config

---

## Citation

Based on:
- **RLOO**: Leave-One-Out Reinforcement Learning for variance reduction
- **Rejection Sampling**: arXiv:2510.13651 §5 (selective training on successes)
- **XSTest**: Paul/XSTest benchmark for overrefusal detection
- **Adversary Prompts**: Multi-strategy jailbreak research (persona, decompose, fictional, academic, analogy)

---

## Next Steps

1. Run GEPA training to see ASR improvement
2. Run XSTest to measure overrefusal vs safety tradeoff
3. Experiment with `--adversary-prompt` variants (which works best?)
4. Try `--rloo-n 5 --target-queries 3` for better signal
5. Compare evolved prompts (GEPA stage 1 attacker instruction) to baseline
