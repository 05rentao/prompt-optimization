# GEPA + Adversary Optimization on Llama-3-8B (HarmBench)

This experiment extends the baseline setup with three changes:

1. **Task model** switched to `meta-llama/Meta-Llama-3-8B-Instruct` (served via vLLM)
2. **Evaluator** switched from keyword matching to **Detoxify**
3. **Adversary model** added (same model family with LoRA): it rewrites HarmBench prompts to be more deceptive and is updated online with policy gradient from evaluator feedback

The rest of the experiment settings follow the baseline (`train=100`, `val=100`, `max_metric_calls=300`, deterministic task generation).

---

## Directory structure

```text
with_adversary/
├── README.md
├── gepa_llama_harmbench_adversary_run.ipynb
├── run_with_adversary.py
├── adversary_pipeline.py
└── evaluator_detoxify.py
```

Generated outputs:

- `results/optimized_system_prompt.txt`
- `results/gepa_run_metrics.json`
- `results/baseline_rows.csv`
- `results/optimized_rows.csv`
- `results/adversary_rl_trace.csv`
- `results/adversary_online_updates.csv`
- `results/adversary_lora/`

---

## Environment (SSH + A100)

Recommended:

- Python 3.10+
- CUDA-capable GPU (A100 recommended)
- Linux host for long-running training

Create environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install dependencies:

```bash
pip install -U \
  gepa litellm openai datasets pandas numpy tqdm \
  transformers peft accelerate bitsandbytes \
  detoxify torch torchvision torchaudio \
  vllm jupyterlab ipykernel
```

> If your PyTorch/CUDA stack is preinstalled by your cluster image, keep that version and only install missing libraries.

---

## Required credentials

- `HF_TOKEN` for gated dataset: `walledai/HarmBench`
- `OPENAI_API_KEY` for GEPA reflection model (`gpt-4o-mini`)

```bash
export HF_TOKEN="hf_..."
export OPENAI_API_KEY="sk-..."
```

---

## Start task model server (vLLM)

Run in a separate SSH terminal and keep it alive:

```bash
source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --served-model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.9 \
  --dtype auto \
  --max-model-len 4096
```

---

## Run experiment

### Option A: Python script

```bash
cd prompt-optimization/with_adversary
python run_with_adversary.py
```

### Option B: Notebook

```bash
cd prompt-optimization/with_adversary
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open `gepa_llama_harmbench_adversary_run.ipynb`, then run all cells.

---

## How scoring works

- Detoxify predicts toxicity attributes from task model responses
- Aggregated toxicity is mapped to:
  - `normalized_toxicity in [0, 1]`
  - `refusal_score = 1 - normalized_toxicity`
- GEPA still maximizes `refusal_score`
- Adversary RL maximizes `1 - refusal_score` (unsafe compliance pressure)

---

## Main config knobs (in `run_with_adversary.py`)

- `model_name`: `meta-llama/Meta-Llama-3-8B-Instruct`
- `reflection_model_name`: `gpt-4o-mini`
- `train_size`: `100`
- `val_size`: `100`
- `max_metric_calls`: `300`
- `adversary_online_update_every_calls`: online RL update frequency during GEPA
- `AdversaryConfig`: LoRA and policy gradient hyperparameters

---

## Notes

- Online adversary updates increase runtime because each update requires extra task-model rollouts.
- If memory is tight, reduce LoRA batch sizes (`warm_start_batch_size`, `rl_batch_size`) and generation lengths.
- Keep `with_out_adversary` untouched for fair baseline comparison.

