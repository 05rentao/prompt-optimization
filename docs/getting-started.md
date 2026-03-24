# Project Getting Started Guide

This is the main onboarding guide for teammates working in this repository.
It consolidates practical run instructions and architecture context that were previously spread across:

- `README.md`
- `getting_started.md`
- `src/README.md`
- `src/runtime/README.md`

Use this file as your first stop before changing code or launching experiments.

## 1) What this project does

This repo studies prompt safety and jailbreak robustness using these **active** run pipelines:

- `runs/gepa_run.py` — GEPA prompt optimization (no adversary training)
- `runs/coev_v2_run.py` — staged CoEV v2 (REINFORCE + dual-role GEPA); main staged pipeline
- `runs/coev_v2_RLOO_run.py` — same staged structure as v2, with RLOO-style adversary updates instead of REINFORCE
- `runs/adversary_run.py` — adversary-only fine-tuning

Legacy (not wired into `scripts/run_unified_experiment.py`): `runs/coev_run.py` (older CoEV modes). Run it directly if you need it.

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

### CoEV v2 (`runs/coev_v2_run.py`) — main staged pipeline

- Stage-based co-evolution: REINFORCE on adversary weights, dual-role GEPA for attacker/defense prompts, optional intra-stage eval, final eval, full artifacts (metrics, CSVs, traces, plots, `run_manifest.json`). Config: `runs.coev_v2` in YAML.

### CoEV v2 RLOO (`runs/coev_v2_RLOO_run.py`)

- Same staged layout and GEPA stages as v2; adversary policy updates use RLOO instead of REINFORCE. Shares `runs.coev_v2` defaults in config.

### Legacy CoEV (`runs/coev_run.py`)

- Older notebook-style CoEV (`reinforce` / `gepa` / `eval`). Not exposed by the unified runner; invoke `uv run runs/coev_run.py ...` directly.

### Adversary-only run (`runs/adversary_run.py`)

- Primary goal: train adversary weights only without GEPA. Use this script to evaluate ASR based on performance of adversary model as a baseline.
- Core idea: baseline eval, adversary REINFORCE loop, final eval.

## 4) Compare and contrast

- Shared: active runs share the same phase ordering (`parse_args` → sessions → data → eval loops → artifacts) and `src/` runtime modules.
- Adversary training: `adversary_run.py`, `coev_v2_run.py` (REINFORCE), `coev_v2_RLOO_run.py` (RLOO); `gepa_run.py` does not train an adversary.
- Prompt evolution: `gepa_run.py`, `coev_v2_run.py`, and `coev_v2_RLOO_run.py` use the GEPA optimization path; legacy `coev_run.py` has its own older GEPA stage wiring.
- Reflection: GEPA-capable runs need an OpenAI-compatible reflection endpoint (`runtime.reflection`), often local vLLM.

## 5) Quick start options

Choose one of these based on your environment and goal.

### Option A: Unified runner script (most common local workflow)

Use `scripts/run_unified_experiment.py` when you want one consistent CLI.

```bash
uv run python scripts/run_unified_experiment.py --mode gepa
uv run python scripts/run_unified_experiment.py --mode coev_v2
uv run python scripts/run_unified_experiment.py --mode coev_v2_rloo
uv run python scripts/run_unified_experiment.py --mode adversary
```

Edit `scripts.unified_runner` in `configs/default.yaml` (or `PROMPT_OPT_CONFIG_PATH`) for `coev_v2_mode`, `coev_v2_rloo_mode`, `adversary_mode`, `run_kind`, result dirs, etc. CLI is only `--mode`.

### Option B: Prime/cluster launcher (recommended on Prime Intellect-style servers)

Use `scripts/launch_unified_prime.sh` for setup + run in one command.

```bash
MODE=gepa bash scripts/launch_unified_prime.sh
MODE=coev_v2 bash scripts/launch_unified_prime.sh
MODE=coev_v2_rloo bash scripts/launch_unified_prime.sh
MODE=adversary bash scripts/launch_unified_prime.sh
```

Experiment settings (dataset sizes, budgets, sub-modes, result paths) live in `configs/default.yaml` or `PROMPT_OPT_CONFIG_PATH` — see `scripts.unified_runner` and `runs.*`.

Notes for Prime usage:

- Ensure GPU drivers/CUDA stack are ready.
- Ensure `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) is exported for gated models/datasets.
- For GEPA and staged CoEV v2 / v2 RLOO, reflection defaults to local vLLM on `127.0.0.1:8001`.

### Option C: Direct run scripts (best for focused debugging)

Defaults come from YAML (`configs/default.yaml` unless overridden); add flags only when you need overrides.

```bash
uv run runs/gepa_run.py
uv run runs/coev_v2_run.py --mode coev
uv run runs/coev_v2_RLOO_run.py --mode coev
uv run runs/adversary_run.py --mode train
# Legacy only:
# uv run runs/coev_run.py --mode reinforce
```

### Option D: Dedicated smoke launcher (no default config edits)

`scripts/launch_smoke_prime.sh` runs short-budget checks using `configs/smoke.yaml` (or `smoke_eval.yaml` when `RUN_KIND=eval`). `MODE=all` runs gepa, coev_v2, coev_v2_rloo, and adversary.

```bash
bash scripts/launch_smoke_prime.sh
RUN_KIND=eval bash scripts/launch_smoke_prime.sh
MODE=coev_v2_rloo bash scripts/launch_smoke_prime.sh
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
uv run runs/gepa_run.py --help
uv run runs/coev_v2_run.py --help
uv run runs/coev_v2_RLOO_run.py --help
uv run runs/adversary_run.py --help

uv run runs/gepa_run.py
uv run runs/coev_v2_run.py --mode coev
uv run runs/coev_v2_RLOO_run.py --mode coev
uv run runs/adversary_run.py --mode train
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
| `runtime.models.adversary_model_id` | Easy swap (within current backend) | Unsloth/HF model id or local path compatible with current Unsloth loader | Used by `coev_v2`, `coev_v2_RLOO`, and `adversary` runs. |
| `runtime.models.target_model_name` | Easy swap | Must match the model served at `runtime.reflection` for HTTP target runs | Same id as `reflection_model_name` by default; victim completions go through vLLM except vector steering. |
| `runtime.models.reflection_model_name` | Easy swap | model id served by the OpenAI-compatible endpoint (`--served-model-name`) | GEPA reflection, CoEV v2, and **target** HTTP generation (shared server). |
| `runtime.models.judge_model_id` | Partial/in progress | Intended: HarmBench judge model id | Documented in config, but current judge construction still uses runtime defaults in scripts. |
| `runtime.reflection.base_url` | Easy swap | OpenAI-compatible endpoint URL | Reflection gateway **and** HTTP target (`build_vllm_target_session`); override with `REFLECTION_VLLM_BASE_URL`. |
| `runtime.reflection.api_key` | Easy swap | endpoint auth token (or `EMPTY` for local vLLM setups) | Same as above; override with `REFLECTION_VLLM_API_KEY`. |
| `runtime.legacy_target_vllm.base_url` | Deprecated/legacy | N/A in current active runs | Kept for legacy compatibility docs; not wired in current pipelines. |
| `runtime.legacy_target_vllm.api_key` | Deprecated/legacy | N/A in current active runs | Kept for legacy compatibility docs; not wired in current pipelines. |
| `runs.gepa.*` | Easy swap | GEPA train/val size, budget, thresholds, prompt, result path | Active and wired in `runs/gepa_run.py`. |
| `runs.coev.*` | Easy swap | Legacy CoEV knobs | Only `runs/coev_run.py`. |
| `runs.coev_v2.*` | Easy swap | CoEV v2 / RLOO schedules, GEPA budgets, token limits, prompt seeds, output names | `runs/coev_v2_run.py` and `runs/coev_v2_RLOO_run.py`. |
| `runs.adversary.*` | Easy swap | adversary training schedule, eval settings, instruction/prompt, output names | Active and wired in `runs/adversary_run.py`. |
| `runs.vector_steering_baseline.target_inference` | Easy swap | `local_hf` (required for steering) | Only `vector_steering_baseline` loads target weights locally; value must stay `local_hf`. |
| `scripts.unified_runner.run_kind` | Easy swap | `train` or `eval` — `eval` forces `coev_v2`, `coev_v2_rloo`, `adversary` sub-modes to eval | `scripts/run_unified_experiment.py`. |
| `scripts.unified_runner.coev_v2_mode` | Easy swap | `coev` or `eval` | CoEV v2 script. |
| `scripts.unified_runner.coev_v2_rloo_mode` | Easy swap | `coev` or `eval` | CoEV v2 RLOO script. |
| `scripts.unified_runner.adversary_mode` | Easy swap | `train` or `eval` | Adversary script. |
| `scripts.unified_runner.save_dir` | Optional | adapter checkpoint root (`--save-dir` when set) | CoEV v2, RLOO, adversary. |
| `scripts.unified_runner.gepa_results_dir` | Easy swap | GEPA output root | Unified runner. |
| `scripts.unified_runner.coev_v2_results_dir` | Easy swap | CoEV v2 output root | Unified runner. |
| `scripts.unified_runner.coev_v2_rloo_results_dir` | Easy swap | CoEV v2 RLOO output root | Unified runner. |
| `scripts.unified_runner.adversary_results_dir` | Easy swap | adversary output root | Unified runner. |
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

- `run_kind`, `coev_v2_mode`, `coev_v2_rloo_mode`, `adversary_mode`: train/eval orchestration for `scripts/run_unified_experiment.py` (CLI: `--mode` only).
- Optional `save_dir`.
- `gepa_results_dir`, `coev_v2_results_dir`, `coev_v2_rloo_results_dir`, `adversary_results_dir`: output roots.
- `runtime_profile`: metadata label (legacy).

## 8) Common artifacts and why they matter

Most runs save a combination of:

- `run_manifest.json`: canonical metadata record (mode, models, dataset, budget, extra fields).
- metrics JSON: compact baseline/final metric summary for dashboards/comparisons.
- CSV outputs: step-level logs, eval rows, optimization traces.
- plots (GEPA / CoEV v2 / v2 RLOO): baseline vs optimized and optimization trajectory.
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

#### `runs/coev_v2_run.py` and `runs/coev_v2_RLOO_run.py`

- Same artifact shape: run config JSON, metrics JSON, CSV bundle (baseline/optimized, training log, traces, stage metrics), plots, `run_manifest.json`, optional `--save-dir` adapters.

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

- `runs/gepa_run.py`
- `runs/coev_v2_run.py`
- `runs/coev_v2_RLOO_run.py`
- `runs/adversary_run.py`

(`runs/coev_run.py` is legacy.)

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

## 11) Streamlining opportunities across runs

- Centralize baseline/final evaluation scaffolding into one helper returning `(metrics, rows)` with a consistent schema.
- Standardize CSV column names across runs (for example `target_response` vs `target_resp`) for easier downstream analysis.
- Normalize CLI names for equivalent knobs (`max_tokens` vs `gepa_max_tokens`) and keep mode naming consistent in manifests.
- Unify artifact writers so every run emits a minimal common contract: `metrics.json`, `baseline.csv`, `final.csv`, and `run_manifest.json`.
- Move reflection endpoint verification/setup into a shared bootstrap helper used by GEPA-capable runs.

## 12) Minimal command cheat sheet

```bash
uv sync

uv run python scripts/run_unified_experiment.py --mode gepa
uv run python scripts/run_unified_experiment.py --mode coev_v2
uv run python scripts/run_unified_experiment.py --mode coev_v2_rloo
uv run python scripts/run_unified_experiment.py --mode adversary

MODE=gepa bash scripts/launch_unified_prime.sh
MODE=coev_v2 bash scripts/launch_unified_prime.sh
MODE=coev_v2_rloo bash scripts/launch_unified_prime.sh
MODE=adversary bash scripts/launch_unified_prime.sh
```
