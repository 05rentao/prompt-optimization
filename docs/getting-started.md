# Project Getting Started Guide

This is the main onboarding guide for teammates working in this repository.
It consolidates practical run instructions and architecture context that were previously spread across:

- `README.md`
- `getting_started.md`
- `src/README.md`
- `src/runtime/README.md`

Use this file as your first stop before changing code or launching experiments.

## 1) What this project does

This repo studies prompt safety and jailbreak robustness using four run pipelines:

- `runs/gepa_run.py` (GEPA prompt optimization)
- `runs/coev_run.py` (legacy CoEV modes)
- `runs/coev_v2_run.py` (staged CoEV + dual-role GEPA)
- `runs/adversary_run.py` (adversary-only fine-tuning)

At a high level, scripts use HarmBench prompts, run target/adversary/judge flows, evaluate ASR/refusal, and save artifacts plus a run manifest.

## 2) Canonical run pipeline

For consistency, run scripts follow the same top-level phase ordering:

1. `parse_args()` + load defaults from `configs/default.yaml`
2. `resolve_device(...)` + build `EvaluationConfig`
3. build long-lived runtime sessions (`RuntimeCatalog`)
4. load data via `load_harmbench_subset(...)` and slice prompts
5. baseline evaluation
6. optimization loop (or eval-only mode)
7. final evaluation
8. save artifacts + `run_manifest.json`

This is the expected pattern when inspecting `runs/`.

## 3) Run types and how they differ

### GEPA run (`runs/gepa_run.py`)

- Primary goal: optimize a defense/system prompt with adversary removed. Use this script to evaluate ASR on GEPA optimized model as a baseline. This is adapted from Mark's notebook (`legacy_code/legacy3/mark_exp.ipynb`)
- Core idea: evaluate baseline prompt, run GEPA reflection-driven optimization, evaluate optimized prompt.

### CoEV run (`runs/coev_run.py`)

- Primary goal: Adapted from Shivs notebook (`legacy_code/legacy3/Copy_of_basic_implementation_without_gepa (4).ipynb`) preserve older old CoEV modes that does not use the GEPA optimize anything API.
- Modes: `reinforce`, `gepa`, `eval`.

### CoEV v2 run (`runs/coev_v2_run.py`) **This is our main pipeline**

- Primary goal: stage-based co-evolution with both weight updates and prompt evolution.
- Implemented pipeline:
  - Start with baseline evaluation using `initial_attacker_instruction` and `initial_defense_prompt`.
  - In each stage, run `iters_per_stage` REINFORCE updates on the adversary weights using sampled train prompts and reward/verdict feedback.
  - Optionally run a stage evaluation every `eval_every_stages` to track pre-evolution ASR/refusal metrics.
  - At the end of each stage, run dual-role GEPA to update both text prompts: attacker instruction and defender system prompt.
  - After all stages, run a final evaluation and persist full artifacts (metrics JSON, eval outputs, training log, attacker/defender GEPA traces, stage metrics, plots, and `run_manifest.json`).

### Adversary-only run (`runs/adversary_run.py`)

- Primary goal: train adversary weights only without GEPA. Use this script to evaluate ASR based on performance of adversary model as a baseline.
- Core idea: baseline eval, adversary REINFORCE loop, final eval.

## 4) Compare and contrast

- Shared: all four runs follow the same high-level phase ordering and use shared `src/` runtime/evaluation modules.
- Adversary training: `adversary_run.py`, `coev_run.py`, and `coev_v2_run.py` run REINFORCE-style updates; `gepa_run.py` does not.
- Prompt evolution: `gepa_run.py` and `coev_v2_run.py` use GEPA optimization; `coev_run.py` implements GEPA from scratch as per Shiv's code `legacy_code/legacy3/Copy_of_basic_implementation_without_gepa (4).ipynb`, using a lighter legacy GEPA stage path.
- Reflection endpoint dependency: GEPA-based runs require an OpenAI-compatible reflection endpoint (`runtime.reflection`), commonly local vLLM.

## 5) Quick start options

Choose one of these based on your environment and goal.

### Option A: Unified runner script (most common local workflow)

Use `scripts/run_unified_experiment.py` when you want one consistent CLI.

```bash
# GEPA
uv run python scripts/run_unified_experiment.py --mode gepa

# CoEV legacy
uv run python scripts/run_unified_experiment.py --mode coev --coev-mode reinforce

# CoEV v2
uv run python scripts/run_unified_experiment.py --mode coev_v2 --coev-v2-mode coev

# Adversary-only
uv run python scripts/run_unified_experiment.py --mode adversary --adversary-mode train
```

### Option B: Prime/cluster launcher (recommended on Prime Intellect-style servers)

Use `scripts/launch_unified_prime.sh` for setup + run in one command.

```bash
# GEPA (starts reflection vLLM, then runs GEPA)
MODE=gepa bash scripts/launch_unified_prime.sh

# CoEV (runs directly, no vLLM startup path)
MODE=coev COEV_MODE=reinforce bash scripts/launch_unified_prime.sh

# CoEV v2 (starts reflection vLLM)
MODE=coev_v2 COEV_V2_MODE=coev bash scripts/launch_unified_prime.sh

# Adversary-only (runs directly)
MODE=adversary ADVERSARY_MODE=train bash scripts/launch_unified_prime.sh
```

Useful launcher overrides:

- `TRAIN_SIZE`, `VAL_SIZE`, `MAX_METRIC_CALLS`, `MAX_TOKENS`
- `EVAL_METHOD`, `REFUSAL_THRESHOLD`, `ASR_THRESHOLD`
- `GEPA_RESULTS_DIR`, `COEV_RESULTS_DIR`

Notes for Prime usage:

- Ensure GPU drivers/CUDA stack are ready.
- Ensure `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) is exported for gated models/datasets.
- For GEPA/CoEV v2, reflection endpoint defaults to local vLLM on `127.0.0.1:8001`.

### Option C: Direct run scripts (best for focused debugging)

```bash
uv run runs/gepa_run.py --show-progress
uv run runs/coev_run.py --mode reinforce
uv run runs/coev_v2_run.py --mode coev
uv run runs/adversary_run.py --mode train
```

### Option D: Dedicated smoke launcher (no default config edits)

Use `scripts/launch_smoke_prime.sh` when you want short-budget sanity checks for one mode or all four modes. It uses `configs/smoke.yaml` by default via `PROMPT_OPT_CONFIG_PATH`, so `configs/default.yaml` remains untouched.

```bash
# Run all four in minimal training mode
bash scripts/launch_smoke_prime.sh

# Run all four in eval-only mode
RUN_KIND=eval bash scripts/launch_smoke_prime.sh

# Run one mode only
MODE=coev bash scripts/launch_smoke_prime.sh
```

## 6) Running locally

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

Before launching long runs, validate that your account/token can access required models:

```bash
# 1) export token
export HF_TOKEN=hf_xxx

# 2) sanity check token works
python3 -c "from huggingface_hub import whoami; print(whoami())"

# 3) verify core model repos used by defaults
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

If a model check fails, confirm:

- your token is valid in the current shell/session
- your Hugging Face account accepted model license/gated terms if required
- model IDs in `configs/default.yaml` match repositories you can access

## 7) Full configuration reference (`configs/default.yaml`)

This section explains each config block in plain language.

### `global`

- `dataset_name`: dataset repository to load (default HarmBench).
- `dataset_config`: dataset subset/config name.
- `dataset_split`: source split to sample from (usually `train`).
- `seed`: random seed for reproducibility.
- `device`: optional explicit device override (`cuda`, `cpu`); `null` means auto-detect.
- `runtime_profile`: high-level runtime tag used in manifests/documentation.

### `runtime.models`

- `adversary_model_id`: trainable adversary model for CoEV/adversary runs.
- `target_model_name`: primary target model for response generation.
- `judge_model_id`: HarmBench classifier model for judge-based ASR scoring.
- `reflection_model_name`: model used by reflection/GEPA prompt optimization.

### `runtime.reflection`

- `base_url`: OpenAI-compatible endpoint for reflection calls.
- `api_key`: auth token for that endpoint (`EMPTY` is common for local vLLM).

### `runtime.legacy_target_vllm`

- Legacy compatibility block for target vLLM endpoint settings.
- Not the primary path for current local target runtime in `runs/gepa_run.py`.

### `runs.gepa`

- `train_size`, `val_size`: HarmBench sample counts for optimization and validation.
- `max_metric_calls`: GEPA evaluator budget.
- `max_tokens`, `temperature`: target generation settings during GEPA/eval.
- `eval_method`: `heuristic` or `judge`.
- `refusal_threshold`, `asr_threshold`: scoring thresholds for refusal/ASR decisions.
- `baseline_system_prompt`: initial defense prompt before optimization.
- `results_dir`: output directory for GEPA artifacts.

### `runs.coev`

- Base controls: `train_size`, `val_size`, `results_dir`, eval thresholds/method.
- `eval_instruction`: default attacker rewrite instruction for eval paths.
- `reinforce.*`:
  - `iterations`, `lr`, `weight_decay`, `eval_every`
  - prompt slices (`train_slice_end`, `eval_slice_start`, `eval_slice_end`)
  - `csv_path` for training logs
- `gepa.*`:
  - staged evolution controls (`stages`, `iters_per_stage`, `eval_every_stages`)
  - optimization hyperparams (`lr`, `weight_decay`)
  - prompt slices + output CSV + initial attacker/defense prompts

### `runs.coev_v2`

- Similar knobs to CoEV but for the v2 staged architecture.
- Adds v2-specific fields:
  - `training_csv_name`
  - `max_new_tokens`
  - GEPA controls (`max_metric_calls`, `gepa_max_tokens`, `gepa_temperature`)
  - initial attacker/defense prompts

### `runs.adversary`

- Controls adversary-only training:
  - dataset sizes, eval method/thresholds
  - `attacker_instruction`, `target_system_prompt`
  - `iterations`, `lr`, `weight_decay`, `eval_every`
  - slice windows and output CSV names
  - `max_new_tokens`
  - `results_dir`

### `scripts.unified_runner`

- `runtime_profile`: label for script profile metadata.
- `gepa_results_dir`, `coev_results_dir`, `adversary_results_dir`: default output roots used by unified runner/launcher.

## 8) Common artifacts and why they matter

Most runs save a combination of:

- `run_manifest.json`: canonical metadata record (mode, models, dataset, budget, extra fields).
- metrics JSON: compact baseline/final metric summary for dashboards/comparisons.
- CSV outputs: step-level logs, eval rows, optimization traces.
- plots (GEPA/CoEV v2): quick visual checks of baseline vs optimized and optimization trajectory.
- optional adapters (`--save-dir`): saved LoRA weights for trained adversary runs.

How to use them:

- compare experiments quickly with metrics JSON and manifest fields
- debug behavior with row-level CSVs (`prompt`, rewritten prompt, response, reward/verdict)
- inspect prompt optimization quality via trace CSV and plots

### What each run writes

#### `runs/adversary_run.py`

- `adversary_run_metrics.json`
- training CSV (`training_csv_name`, default from `configs/default.yaml`)
- eval CSV (`eval_csv_name`, default from `configs/default.yaml`)
- `run_manifest.json`
- optional adapter directory from `--save-dir`

#### `runs/coev_run.py`

- REINFORCE CSV (`reinforce.csv_path`) when `--mode reinforce`
- GEPA CSV (`gepa.csv_path`) when `--mode gepa`
- `run_manifest.json`
- optional adapter directory from `--save-dir`
- note: CSV paths come from config and may be outside `--results-dir` if configured that way

#### `runs/coev_v2_run.py`

- run metadata JSON (`coev_v2_run_config.json`)
- metrics JSON (`coev_v2_metrics.json`)
- CSV bundle from `write_many_csv(...)` (baseline/optimized outputs, training log, traces, stage metrics)
- baseline-vs-optimized bar plot
- optimizer trajectory plot(s)
- `run_manifest.json`
- optional adapter directory from `--save-dir`

#### `runs/gepa_run.py`

- best prompt text file (`optimized_system_prompt.txt`)
- metrics JSON (`gepa_metrics.json`)
- CSV bundle (comparison, baseline outputs, optimized outputs)
- optional `optimizer_trace.csv` when trace rows exist
- baseline-vs-optimized bar plot
- optimizer trajectory plot
- `run_manifest.json`

## 9) `src/` and `src/runtime/` orientation

Top-level responsibilities:

- `runs/` owns orchestration: CLI args, phase ordering, logging, artifact naming.
- `src/` owns reusable implementation: data loading, scoring, artifact helpers, runtime wrappers, GEPA helpers.
- `src/runtime/` owns backend-specific adapters behind stable interfaces.

Primary consumer scripts:

- `runs/coev_run.py`
- `runs/coev_v2_run.py`
- `runs/gepa_run.py`
- `runs/adversary_run.py`

Main modularity components:

- `src/runtime/`
  - runtime session builders (`RuntimeCatalog`)
  - target/adversary/judge/reflection implementations
  - shared evaluation (`evaluate_outputs`, `evaluate_examples`)
  - GEPA optimization helpers
- `src/data.py`
  - HarmBench subset loading and split shaping
- `src/artifacts.py`
  - JSON/CSV writers, plots, and manifest writer
- `src/evaluators.py`
  - refusal heuristics + judge verdict normalization helpers
- `src/run_pipeline.py`
  - shared orchestration helpers across runs

## 10) Suggested teammate workflow for making changes

1. Pick one run script as the target behavior.
2. Keep reusable logic in `src/`, not in `runs/`.
3. Keep run-specific control flow in `runs/`.
4. Update docs when changing CLI, mode names, or artifact contracts:
   - `README.md`
   - `docs/getting-started.md`
   - `src/README.md` and/or `src/runtime/README.md` if internal APIs changed
5. Run a short-budget smoke test before large Prime jobs.

## 11) Streamlining opportunities across all four runs

- Centralize baseline/final evaluation scaffolding into one helper returning `(metrics, rows)` with a consistent schema.
- Standardize CSV column names across runs (for example `target_response` vs `target_resp`) for easier downstream analysis.
- Normalize CLI names for equivalent knobs (`max_tokens` vs `gepa_max_tokens`) and keep mode naming consistent in manifests.
- Unify artifact writers so every run emits a minimal common contract: `metrics.json`, `baseline.csv`, `final.csv`, and `run_manifest.json`.
- Move reflection endpoint verification/setup into a shared bootstrap helper used by GEPA-capable runs.

## 12) Minimal command cheat sheet

```bash
# Install dependencies
uv sync

# Unified runner
uv run python scripts/run_unified_experiment.py --mode gepa
uv run python scripts/run_unified_experiment.py --mode coev --coev-mode reinforce
uv run python scripts/run_unified_experiment.py --mode coev_v2 --coev-v2-mode coev
uv run python scripts/run_unified_experiment.py --mode adversary --adversary-mode train

# Prime launcher
MODE=gepa bash scripts/launch_unified_prime.sh
MODE=coev bash scripts/launch_unified_prime.sh
MODE=coev_v2 bash scripts/launch_unified_prime.sh
MODE=adversary bash scripts/launch_unified_prime.sh
```
