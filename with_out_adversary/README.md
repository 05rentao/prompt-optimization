# GEPA Prompt Optimization on Qwen2.5-3B (HarmBench)

This repository contains a Jupyter notebook that runs one full **GEPA** (Reflective Text Evolution) optimization cycle to improve the system prompt of a local **Qwen2.5-3B-Instruct** model for safety behavior, evaluated on a subset of the **HarmBench** dataset.

---

## What the notebook produces

- Baseline vs. optimized system prompt comparison (metrics + plots)
- Optimization trajectory chart (score over evaluator calls)
- Exported artifacts: `optimized_system_prompt.txt`, `gepa_run_metrics.json`, CSVs in `results/`

---

## Hardware requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA GPU with ≥ 16 GB VRAM (A100 recommended) |
| RAM | ≥ 32 GB system RAM |
| Disk | ≥ 20 GB free (model weights + venv) |
| OS | Linux (Ubuntu 20.04+ recommended) |

---

## Prerequisites

### 1. Python environment

Python **3.10 or newer** is required. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

The notebook's dependency cell handles this automatically, but you can also install upfront:

```bash
pip install -U gepa openai>=1.40.0 datasets>=2.20.0 \
    pandas>=2.2.0 numpy>=1.26.0 \
    matplotlib>=3.8.0 seaborn>=0.13.0 tqdm>=4.66.0 \
    litellm
```

> **Note:** `litellm` is an internal GEPA dependency for routing LLM API calls. It must be installed in the same environment.

### 3. Install vLLM

vLLM is needed to serve the Qwen2.5-3B-Instruct model locally:

```bash
pip install vllm
```

Verify the installation:

```bash
python -c "import vllm; print('vLLM OK')"
```

### 4. Download the task model

The notebook uses **Qwen/Qwen2.5-3B-Instruct** from Hugging Face.  
The model will be downloaded automatically when the vLLM server starts (step 6 below).  
Alternatively, pre-download it manually:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

### 5. Obtain required credentials

You need **two** credentials before running the notebook:

#### a) OpenAI API key (`OPENAI_API_KEY`)

Used by GEPA's reflection model (`gpt-4o-mini`) to propose improved system prompts.

- Create a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Ensure the key has access to `gpt-4o-mini`

Export it in your shell **before** launching Jupyter:

```bash
export OPENAI_API_KEY="sk-..."
```

Alternatively the notebook will prompt you to enter it interactively (hidden input).

> **Security:** Never hardcode this key in the notebook. It is read from the environment variable and is excluded from git via `.gitignore`.

#### b) Hugging Face token (`HF_TOKEN`)

Required because `walledai/HarmBench` is a **gated dataset**.

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Visit [huggingface.co/datasets/walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench) and request access (usually approved instantly)
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is sufficient)

Export it in your shell:

```bash
export HF_TOKEN="hf_..."
```

The notebook also accepts it via interactive prompt if the variable is not set.

### 6. Start the vLLM server

In a **separate terminal** (keep it running while the notebook executes):

```bash
source .venv/bin/activate   # activate the same environment
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --served-model-name Qwen/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --max-model-len 4096
```

Wait until you see a line like:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The notebook connectivity check cell will fail with a clear error message if the server is not reachable.

### 7. Install and start Jupyter

```bash
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open the URL printed in the terminal (contains a token), then open `gepa_qwen_harmbench_run.ipynb`.

**Kernel selection:** Choose the Python interpreter from the virtual environment:

```
.venv/bin/python
```

If it does not appear in the kernel list, register it first:

```bash
pip install ipykernel
python -m ipykernel install --user --name gepa-venv --display-name "GEPA (.venv)"
```

Then select **GEPA (.venv)** from the Jupyter kernel menu.

---

## Running the notebook

Run all cells top-to-bottom (**Kernel → Restart & Run All** is the cleanest approach).

Expected execution order:

| Cell | Action |
|------|--------|
| Config | Sets experiment parameters (model names, dataset size, GEPA budget) |
| Credentials | Reads / prompts for `OPENAI_API_KEY` and `HF_TOKEN` |
| Install deps | Installs Python packages via pip |
| Imports | Loads libraries |
| Load dataset | Downloads 100 train + 100 val examples from HarmBench |
| vLLM check | Verifies server is up and model is loaded |
| Baseline eval | Evaluates the initial system prompt |
| GEPA run | Runs ≈ 300 evaluator calls (~3–4 optimization iterations) |
| Results | Prints metrics, renders plots, saves artifacts |

> **Runtime estimate:** With an A100, the full 300-call run takes roughly 30–60 minutes depending on network latency to the OpenAI API.

---

## Key configuration knobs

All tunable parameters are at the top of the first code cell:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | Local task model served by vLLM |
| `REFLECTION_MODEL_NAME` | `gpt-4o-mini` | OpenAI model used for GEPA reflection |
| `TRAIN_SIZE` | `100` | Training examples used during optimization |
| `VAL_SIZE` | `100` | Validation examples used for scoring |
| `MAX_METRIC_CALLS` | `300` | Total evaluator budget (≈ iterations × VAL_SIZE) |
| `MAX_TOKENS` | `256` | Max tokens per model response |
| `TEMPERATURE` | `0.0` | Deterministic generation |

---

## Evaluation metrics

| Metric | Range | Better when |
|--------|-------|-------------|
| `refusal_rate` | [0, 1] | Higher — fraction of responses that refuse harmful requests |
| `unsafe_compliance_rate` | [0, 1] | Lower — fraction of responses that comply with harmful requests |
| `mean_refusal_score` | [0, 1] | Higher — mean per-example score (0 = comply, 0.7 = partial, 1.0 = full refusal) |
| `aggregate_score` | [0, 1] | Higher — primary GEPA objective (equals `mean_refusal_score`) |

---

## Repository structure

```
GEPA-Qwen/
├── gepa_qwen_harmbench_run.ipynb   # Main experiment notebook
├── .gitignore                       # Excludes .venv, results, secrets
└── README.md                        # This file

# Generated at runtime (not tracked by git):
├── optimized_system_prompt.txt
├── gepa_run_metrics.json
└── results/
    ├── baseline_vs_optimized_metrics.csv
    ├── optimizer_trace.csv
    ├── plot_baseline_vs_optimized.png
    └── plot_optimization_trajectory.png
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Cannot reach local vLLM endpoint` | Make sure the vLLM server is running in a separate terminal on port 8000 |
| `DatasetNotFoundError: gated dataset` | Visit the HarmBench dataset page on HF and request access; ensure `HF_TOKEN` is set |
| `ModuleNotFoundError: No module named 'litellm'` | Run `pip install litellm` in the same venv |
| `ModuleNotFoundError: No module named 'vllm'` | Run `pip install vllm` in the same venv |
| `RuntimeError: OPENAI_API_KEY is empty` | Export the key: `export OPENAI_API_KEY="sk-..."` or enter it in the credentials cell |
| Kernel uses wrong Python | Re-select the `.venv` kernel; register it with `python -m ipykernel install --user` |
