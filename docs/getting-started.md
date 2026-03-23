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
uv run python scripts/run_unified_experiment.py --mode coev

# CoEV v2
uv run python scripts/run_unified_experiment.py --mode coev_v2

# Adversary-only
uv run python scripts/run_unified_experiment.py --mode adversary
```

Edit `scripts.unified_runner` in `configs/default.yaml` for `coev_mode`, `coev_v2_mode`, `adversary_mode`, `run_kind`, output dirs, etc.

### Option B: Prime/cluster launcher (recommended on Prime Intellect-style servers)

Use `scripts/launch_unified_prime.sh` for setup + run in one command.

```bash
# GEPA (starts reflection vLLM, then runs GEPA)
MODE=gepa bash scripts/launch_unified_prime.sh

# CoEV
MODE=coev bash scripts/launch_unified_prime.sh

# CoEV v2 (starts reflection vLLM)
MODE=coev_v2 bash scripts/launch_unified_prime.sh

# Adversary-only
MODE=adversary bash scripts/launch_unified_prime.sh
```

Experiment settings (dataset sizes, budgets, sub-modes, result paths) live in `configs/default.yaml` or `PROMPT_OPT_CONFIG_PATH` — see `scripts.unified_runner` and `runs.*`.

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
    "meta-llama/Llama-3.1-8B-Instruct",
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

### 7.1 Swapability status (what is truly config-driven)

Legend:

- Easy swap: changing the config value changes runtime behavior directly.
- Partial/in progress: key exists and is read, but behavior is limited or only used in metadata.
- Deprecated/legacy: key exists for compatibility/docs but is not part of the active runtime path.

| Config path | Status | What you can swap | Notes |
|---|---|---|---|
| `global.dataset_name` | Easy swap | dataset repo id (for example `walledai/HarmBench`) | Used by all run scripts. |
| `global.dataset_config` | Easy swap | dataset subset/config name | Used by all run scripts. |
| `global.dataset_split` | Easy swap | split name (for example `train`, `test`) | Used by all run scripts. |
| `global.seed` | Easy swap | random seed integer | Used by all run scripts. |
| `global.device` | Easy swap | `cuda`, `cpu`, or `null` for auto | Passed into runtime device resolution. |
| `global.runtime_profile` | Partial/in progress | label string (for manifests/reporting) | Currently metadata/profile label, not a backend selector. |
| `runtime.models.adversary_model_id` | Easy swap (within current backend) | Unsloth/HF model id or local path compatible with current Unsloth loader | Used by `coev`, `coev_v2`, and `adversary` runs. |
| `runtime.models.target_model_name` | Easy swap | Must match the model served at `runtime.reflection` for HTTP target runs | Same id as `reflection_model_name` by default; victim completions go through vLLM except vector steering. |
| `runtime.models.reflection_model_name` | Easy swap | model id served by the OpenAI-compatible endpoint (`--served-model-name`) | GEPA reflection, CoEV v2, and **target** HTTP generation (shared server). |
| `runtime.models.judge_model_id` | Partial/in progress | Intended: HarmBench judge model id | Documented in config, but current judge construction still uses runtime defaults in scripts. |
| `runtime.reflection.base_url` | Easy swap | OpenAI-compatible endpoint URL | Reflection gateway **and** HTTP target (`build_vllm_target_session`); override with `REFLECTION_VLLM_BASE_URL`. |
| `runtime.reflection.api_key` | Easy swap | endpoint auth token (or `EMPTY` for local vLLM setups) | Same as above; override with `REFLECTION_VLLM_API_KEY`. |
| `runtime.legacy_target_vllm.base_url` | Deprecated/legacy | N/A in current active runs | Kept for legacy compatibility docs; not wired in current pipelines. |
| `runtime.legacy_target_vllm.api_key` | Deprecated/legacy | N/A in current active runs | Kept for legacy compatibility docs; not wired in current pipelines. |
| `runs.gepa.*` | Easy swap | GEPA train/val size, budget, thresholds, prompt, result path | Active and wired in `runs/gepa_run.py`. |
| `runs.coev.*` | Easy swap | CoEV modes, schedules, eval settings, prompt seeds, csv paths | Active and wired in `runs/coev_run.py`. |
| `runs.coev_v2.*` | Easy swap | CoEV v2 schedules, GEPA budgets, token limits, prompt seeds, output names | Active and wired in `runs/coev_v2_run.py`. |
| `runs.adversary.*` | Easy swap | adversary training schedule, eval settings, instruction/prompt, output names | Active and wired in `runs/adversary_run.py`. |
| `runs.vector_steering_baseline.target_inference` | Easy swap | `local_hf` (required for steering) | Only `vector_steering_baseline` loads target weights locally; value must stay `local_hf`. |
| `scripts.unified_runner.run_kind` | Easy swap | `train` or `eval` — `eval` forces coev / coev_v2 / adversary pipeline modes to eval | Read by `scripts/run_unified_experiment.py`. |
| `scripts.unified_runner.coev_mode` | Easy swap | `reinforce`, `gepa`, or `eval` (CoEV legacy) | Same. |
| `scripts.unified_runner.coev_v2_mode` | Easy swap | `coev` or `eval`; omit/`null` to infer from `coev_mode` | Same. |
| `scripts.unified_runner.adversary_mode` | Easy swap | `train` or `eval` | Same. |
| `scripts.unified_runner.gepa_show_progress` | Easy swap | progress bar for GEPA | Same. |
| `scripts.unified_runner.save_dir` | Optional | adapter checkpoint root (passed as `--save-dir` when set) | Same. |
| `scripts.unified_runner.gepa_results_dir` | Easy swap | default GEPA output root | Used by unified runner. |
| `scripts.unified_runner.coev_results_dir` | Easy swap | default CoEV output root | Used by unified runner. |
| `scripts.unified_runner.coev_v2_results_dir` | Easy swap | CoEV v2 output root | Used by unified runner. |
| `scripts.unified_runner.adversary_results_dir` | Easy swap | default adversary output root | Used by unified runner. |
| `scripts.unified_runner.runtime_profile` | Deprecated/legacy | N/A in current launcher logic | Present in config/docs but not used by `scripts/run_unified_experiment.py`. |

### 7.2 Practical interpretation

- Target inference for most runs is **HTTP** to the same vLLM process as GEPA reflection (`OpenAIChatTargetRuntime`). Launch scripts set `REFLECTION_VLLM_BASE_URL` to point Python at the server.
- **Vector steering** is the exception: it requires local weights and uses `runs.vector_steering_baseline.target_inference: local_hf`.
- Keys in `runtime.legacy_target_vllm.*` and `scripts.unified_runner.runtime_profile` remain legacy placeholders for older docs/workflows.

### `global`

- `dataset_name`: dataset repository to load (default HarmBench).
- `dataset_config`: dataset subset/config name.
- `dataset_split`: source split to sample from (usually `train`).
- `seed`: random seed for reproducibility.
- `device`: optional explicit device override (`cuda`, `cpu`); `null` means auto-detect.
- `runtime_profile`: high-level runtime tag used in manifests/documentation.

### `runtime.models`

- `adversary_model_id`: trainable adversary model for CoEV/adversary runs.
- `target_model_name`: victim model id (must match the served vLLM model id for HTTP target runs; same as `reflection_model_name` in defaults).
- `judge_model_id`: HarmBench classifier model for judge-based ASR scoring.
- `reflection_model_name`: model used by reflection/GEPA prompt optimization.

### `runtime.reflection`

- `base_url`: OpenAI-compatible endpoint for reflection calls.
- `api_key`: auth token for that endpoint (`EMPTY` is common for local vLLM).

### `runtime.legacy_target_vllm`

- Legacy compatibility block; active pipelines use `runtime.reflection` for the shared vLLM URL.

### `runs.vector_steering_baseline`

- `target_inference`: must be `local_hf` so the script can read model weights for steering vectors.

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

- `run_kind`, `coev_mode`, `coev_v2_mode`, `adversary_mode`: orchestration for `scripts/run_unified_experiment.py` (CLI is only `--mode`).
- `gepa_show_progress`, `save_dir` (optional).
- `gepa_results_dir`, `coev_results_dir`, `coev_v2_results_dir`, `adversary_results_dir`: output roots for the unified runner.
- `runtime_profile`: label for script profile metadata (legacy / docs).

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

# Unified runner (edit scripts.unified_runner for sub-modes and paths)
uv run python scripts/run_unified_experiment.py --mode gepa
uv run python scripts/run_unified_experiment.py --mode coev
uv run python scripts/run_unified_experiment.py --mode coev_v2
uv run python scripts/run_unified_experiment.py --mode adversary

# Prime launcher
MODE=gepa bash scripts/launch_unified_prime.sh
MODE=coev bash scripts/launch_unified_prime.sh
MODE=coev_v2 bash scripts/launch_unified_prime.sh
MODE=adversary bash scripts/launch_unified_prime.sh
```
