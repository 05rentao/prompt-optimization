# STAT 4830 — Prompt Optimization Runs

This repository contains four experiment runners under `runs/` that share a common flow:
an adversary rewrites harmful prompts, a target model responds, and evaluation tracks ASR/refusal.  
Some runs only train adversary weights, while others optimize attacker/defense prompts with GEPA.

The reusable runtime/evaluation/artifact logic lives in `src/`. See `src/README.md` for the detailed module guide.

## Canonical run pipeline

For consistency, the run scripts are organized in the same top-level phase order:

1. `parse_args()` + load defaults from `configs/default.yaml`
2. `resolve_device(...)` + build `EvaluationConfig`
3. build long-lived runtime sessions (`RuntimeCatalog`)
4. load data via `load_harmbench_subset(...)` and slice prompts
5. baseline evaluation
6. optimization loop (or eval-only mode)
7. final evaluation
8. save artifacts + `run_manifest.json`

This is the pattern to look for when inspecting code in `runs/`.

## Runs Overview

### `runs/adversary_run.py`
- **Purpose:** adversary-only REINFORCE fine-tuning (no prompt optimization).
- **Pipeline shape:** full baseline -> train loop -> final eval -> artifacts.
- **Unique behavior:** updates only adversary weights; target system prompt stays fixed.
- **Artifacts:** metrics JSON + training CSV + eval CSV + manifest (+ optional LoRA adapter save).

### `runs/coev_run.py`
- **Purpose:** legacy CoEV runner with `reinforce`, `gepa` implmentation from scratch, or `eval` mode.
- **Pipeline shape:** baseline eval first, then mode-specific optimize/eval path, then manifest.
- **Unique behavior:** using GEPA evolution implmented from scratch (reflection from an LLM), while preserving REINFORCE mode.
- **Artifacts:** mode CSV logs + manifest (+ optional LoRA adapter save).

### `runs/coev_v2_run.py`
- **Purpose:** staged CoEV pipeline combining REINFORCE updates with dual-role GEPA.
- **Pipeline shape:** baseline eval -> staged training/evolution -> final eval -> full artifact bundle.
- **Unique behavior:** alternates adversary weight updates with attacker/defender GEPA prompt evolution per stage.
- **Artifacts:** metrics JSON, comparison/trace/stage CSVs, plots, manifest (+ optional LoRA adapter save).

### `runs/gepa_run.py`
- **Purpose:** GEPA-only system prompt optimization for defense behavior.
- **Pipeline shape:** baseline eval -> GEPA optimization -> final eval -> artifacts.
- **Unique behavior:** no adversary weight updates; focuses on improving a defense/system prompt.
- **Artifacts:** metrics JSON, baseline/optimized CSVs, optimizer trace, plots, manifest.

## Compare and contrast

- **Shared:** all four runs now follow the same high-level phase ordering and use shared `src/` runtime/evaluation modules.
- **Adversary training:** `adversary_run.py`, `coev_run.py`, and `coev_v2_run.py` run REINFORCE-style updates; `gepa_run.py` does not.
- **Prompt evolution:** `gepa_run.py` and `coev_v2_run.py` use GEPA optimization; `coev_run.py` has a lighter legacy GEPA stage path.
- **Reflection endpoint dependency:** GEPA-based runs require an OpenAI-compatible reflection endpoint (`runtime.reflection` in config), commonly a local vLLM service.
- **Persistence depth:** `coev_v2_run.py` and `gepa_run.py` save the richest artifact sets (plots + multiple traces), while `coev_run.py` stays minimal.

## Streamlining opportunities across all four runs

- Centralize baseline/final evaluation scaffolding into one helper that returns `(metrics, rows)` with a consistent schema.
- Standardize CSV column names across runs (for example `target_response` vs `target_resp`) for easier downstream analysis.
- Normalize CLI names for equivalent knobs (`max_tokens` vs `gepa_max_tokens`) and keep mode naming consistent in manifests.
- Unify artifact writers so every run emits a minimal common contract: `metrics.json`, `baseline.csv`, `final.csv`, and `run_manifest.json`.
- Move reflection endpoint verification/setup into a shared bootstrap helper used by GEPA-capable runs.

## Artifact generation details

### Common artifact flow

Across runs, artifact generation happens after baseline/final metrics are available:

1. build metrics payloads (baseline/final stats + run metadata)
2. write structured tables (CSV) for analysis/debugging
3. write `run_manifest.json` as the index of run settings and outputs
4. optionally save adapters (adversary-capable runs)
5. optionally save plots/extra trace files (GEPA-focused runs)

By default, artifacts are written under each run's `--results-dir` (or that run's config default), except where noted below.

### What each run writes

### `runs/adversary_run.py`
- `adversary_run_metrics.json`
- training CSV (`training_csv_name`, default from `configs/default.yaml`)
- eval CSV (`eval_csv_name`, default from `configs/default.yaml`)
- `run_manifest.json`
- optional adapter directory from `--save-dir`

### `runs/coev_run.py`
- REINFORCE CSV (`reinforce.csv_path`) when `--mode reinforce`
- GEPA CSV (`gepa.csv_path`) when `--mode gepa`
- `run_manifest.json`
- optional adapter directory from `--save-dir`
- note: the CSV paths come from config and may be outside `--results-dir` if configured that way

### `runs/coev_v2_run.py`
- run metadata JSON (`coev_v2_run_config.json`)
- metrics JSON (`coev_v2_metrics.json`)
- CSV bundle from `write_many_csv(...)` (baseline/optimized outputs, training log, traces, stage metrics)
- baseline-vs-optimized bar plot
- optimizer trajectory plot(s)
- `run_manifest.json`
- optional adapter directory from `--save-dir`

### `runs/gepa_run.py`
- best prompt text file (`optimized_system_prompt.txt`)
- metrics JSON (`gepa_metrics.json`)
- CSV bundle (comparison, baseline outputs, optimized outputs)
- optional `optimizer_trace.csv` when trace rows exist
- baseline-vs-optimized bar plot
- optimizer trajectory plot
- `run_manifest.json`

## Getting started

Use one of these two workflows depending on your environment.

### Option A: Prime launcher (recommended on cluster/GPU server)

```bash
# GEPA (starts reflection vLLM, then runs unified runner in gepa mode)
MODE=gepa bash scripts/launch_unified_prime.sh

# CoEV (no vLLM startup path)
MODE=coev COEV_MODE=reinforce bash scripts/launch_unified_prime.sh

# CoEV v2 (starts reflection vLLM, then runs coev_v2 mode)
MODE=coev_v2 COEV_V2_MODE=coev bash scripts/launch_unified_prime.sh

# Adversary-only (no vLLM startup path)
MODE=adversary ADVERSARY_MODE=train bash scripts/launch_unified_prime.sh
```

Useful overrides:
- `TRAIN_SIZE`, `VAL_SIZE`, `MAX_METRIC_CALLS`, `MAX_TOKENS`
- `EVAL_METHOD`, `REFUSAL_THRESHOLD`, `ASR_THRESHOLD`
- `GEPA_RESULTS_DIR`, `COEV_RESULTS_DIR`

### Option B: Run scripts directly

```bash
# Setup once
uv sync

# Run a single pipeline directly
uv run runs/gepa_run.py --show-progress
uv run runs/coev_run.py --mode reinforce
uv run runs/coev_v2_run.py --mode coev
uv run runs/adversary_run.py --mode train
```

Use direct script calls when developing/debugging one pipeline; use the launcher when you want repeatable server setup and runtime orchestration.

## Running locally

### Prerequisites

- Python 3.10+
- GPU recommended for local model + judge workloads
- `uv` package manager: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)

### Setup

```bash
uv sync
```

### Running

```bash
# Inspect script options
uv run runs/adversary_run.py --help
uv run runs/coev_run.py --help
uv run runs/coev_v2_run.py --help
uv run runs/gepa_run.py --help

# Typical runs
uv run runs/adversary_run.py --mode train
uv run runs/coev_run.py --mode reinforce
uv run runs/coev_v2_run.py --mode coev
uv run runs/gepa_run.py --show-progress
```

### Notes on keys/endpoints

- HarmBench dataset loading may require Hugging Face authentication depending on environment (`HF_TOKEN`).
- GEPA-based runs need an OpenAI-compatible reflection endpoint configured in `configs/default.yaml`.
- A paid external API key is not required when using local vLLM + `api_key: EMPTY`.

### Verify Hugging Face model access

Before launching long runs, double check that your account/token can access the required models.

```bash
# 1) export token
export HF_TOKEN=hf_xxx

# 2) Sanity check token works
python3 -c "from huggingface_hub import whoami; print(whoami())"

# 3) Verify you can resolve the core model repos used by defaults
python3 - <<'PY'
from huggingface_hub import model_info
models = [
    "unsloth/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "cais/HarmBench-Mistral-7b-val-cls",
]
for m in models:
    try:
        model_info(m)
        print(f"OK: {m}")
    except Exception as e:
        print(f"FAIL: {m} -> {e}")
PY
```

If a model check fails, make sure:
- your token is valid in the current shell/session,
- your Hugging Face account has accepted the model's license/gated terms (if applicable),
- the model IDs in `configs/default.yaml` match repositories you can access.

## License / course

For STAT 4830 use. Adversarial and harmful behavior data are for research and course purposes only.