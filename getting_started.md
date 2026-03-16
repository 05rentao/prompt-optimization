# Project Getting Started Guide

This is the main onboarding guide for teammates working in this repository.  
It combines the practical run instructions and architecture context from:
- `README.md`
- `src/README.md`
- `src/runtime/README.md`

Use this file as your first stop before changing code or launching experiments.

## 1) What this project does

This repo studies prompt safety and jailbreak robustness using four run pipelines:
- `runs/gepa_run.py` (GEPA prompt optimization)
- `runs/coev_run.py` (legacy CoEV modes)
- `runs/coev_v2_run.py` (staged CoEV + dual-role GEPA)
- `runs/adversary_run.py` (adversary-only fine-tuning)

At a high level, scripts use HarmBench prompts, run target/adversary/judge flows, evaluate ASR/refusal, and save artifacts + a run manifest.

---

## 2) Run types and how they differ

### GEPA run (`runs/gepa_run.py`)
- **Primary goal:** optimize a defense/system prompt.
- **Core idea:** evaluate baseline prompt -> run GEPA reflection-driven optimization -> evaluate optimized prompt.
- **Unique traits:** no adversary weight updates; strongest focus on prompt search.

### CoEV run (`runs/coev_run.py`)
- **Primary goal:** keep the older flexible CoEV modes alive.
- **Modes:** `reinforce`, `gepa`, `eval`.
- **Unique traits:** simpler/legacy training/evolution path; lighter artifact set.

### CoEV v2 run (`runs/coev_v2_run.py`)
- **Primary goal:** stage-based co-evolution with both weight updates and prompt evolution.
- **Core idea:** REINFORCE updates during stages, then dual-role GEPA boundary updates.
- **Unique traits:** richest combined training + prompt optimization flow.

### Adversary-only run (`runs/adversary_run.py`)
- **Primary goal:** train adversary weights only.
- **Core idea:** baseline eval -> adversary REINFORCE loop -> final eval.
- **Unique traits:** no prompt optimization, no reflection loop.

---

## 3) Quick start options

Choose one of these based on your environment and goal.

## Option A: Unified runner script (most common local workflow)

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

## Option B: Prime/cluster launcher (recommended on Prime Intellect-style servers)

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

Notes for Prime usage:
- Make sure GPU drivers/CUDA stack are ready.
- Ensure `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) is exported for gated models/datasets.
- For GEPA/CoEV v2, reflection endpoint defaults to local vLLM on `127.0.0.1:8001`.

## Option C: Direct run scripts (best for focused debugging)

```bash
uv run runs/gepa_run.py --show-progress
uv run runs/coev_run.py --mode reinforce
uv run runs/coev_v2_run.py --mode coev
uv run runs/adversary_run.py --mode train
```

## Option D: Dedicated smoke launcher (no default config edits)

Use `scripts/launch_smoke_prime.sh` when you want short-budget sanity checks
for one mode or all four modes. It uses `configs/smoke.yaml` by default via
`PROMPT_OPT_CONFIG_PATH`, so your `configs/default.yaml` stays untouched.

```bash
# Run all four in minimal training mode
bash scripts/launch_smoke_prime.sh

# Run all four in eval-only mode
RUN_KIND=eval bash scripts/launch_smoke_prime.sh

# Run one mode only
MODE=coev bash scripts/launch_smoke_prime.sh
```

---

## 4) Setup checklist (before first run)

1. Install deps:
   ```bash
   uv sync
   ```
2. Export HF token (if needed):
   ```bash
   export HF_TOKEN=hf_xxx
   ```
3. Verify token:
   ```bash
   python3 -c "from huggingface_hub import whoami; print(whoami())"
   ```
4. If running GEPA/CoEV v2 on Prime, verify reflection vLLM endpoint startup (launcher handles this).

---

## 5) Full configuration reference (`configs/default.yaml`)

This section explains each config block in plain language.

## `global`

- `dataset_name`: dataset repository to load (default HarmBench).
- `dataset_config`: dataset subset/config name (for HF datasets with named configs).
- `dataset_split`: source split to sample from (usually `train`).
- `seed`: random seed for sampling/shuffling reproducibility.
- `device`: optional explicit device override (`cuda`, `cpu`); `null` means auto-detect.
- `runtime_profile`: high-level runtime tag used in manifests/documentation (`local_transformers` by default).

## `runtime.models`

- `adversary_model_id`: trainable adversary model used in CoEV/adversary runs.
- `target_model_name`: primary target model for response generation.
- `judge_model_id`: HarmBench classifier model for judge-based ASR scoring.
- `reflection_model_name`: model used by reflection/GEPA prompt optimization.

## `runtime.reflection`

- `base_url`: OpenAI-compatible endpoint for reflection calls (typically local vLLM).
- `api_key`: auth token for that endpoint (`EMPTY` is common for local vLLM).

## `runtime.legacy_target_vllm`

- Legacy compatibility block for target vLLM endpoint settings.
- Not the primary path for current local target runtime in `runs/gepa_run.py`.

## `runs.gepa`

- `train_size`, `val_size`: HarmBench sample counts for optimization and validation.
- `max_metric_calls`: GEPA budget (how many evaluator calls optimization can spend).
- `max_tokens`, `temperature`: target generation settings during GEPA/eval.
- `eval_method`: `heuristic` or `judge`.
- `refusal_threshold`, `asr_threshold`: scoring thresholds for refusal/ASR decisions.
- `baseline_system_prompt`: initial defense prompt before optimization.
- `results_dir`: output directory for GEPA artifacts.

## `runs.coev`

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

## `runs.coev_v2`

- Similar training/eval knobs to CoEV but for the v2 staged architecture.
- Adds v2-specific fields:
  - `training_csv_name`
  - `max_new_tokens`
  - GEPA controls (`max_metric_calls`, `gepa_max_tokens`, `gepa_temperature`)
  - initial attacker/defense prompts

## `runs.adversary`

- Controls adversary-only training:
  - dataset sizes, eval method/thresholds
  - `attacker_instruction`, `target_system_prompt`
  - `iterations`, `lr`, `weight_decay`, `eval_every`
  - slice windows and output CSV names
  - `max_new_tokens`
  - `results_dir`

## `scripts.unified_runner`

- `runtime_profile`: label for script profile metadata.
- `gepa_results_dir`, `coev_results_dir`, `adversary_results_dir`: default output roots used by unified runner/launcher.

---

## 6) Common artifacts and why they matter

Most runs save a combination of:
- `run_manifest.json`: canonical metadata record (mode, models, dataset, budget, extra fields).
- metrics JSON: compact baseline/final metric summary for dashboards/comparisons.
- CSV outputs: step-level logs, eval rows, optimization traces.
- plots (GEPA/coev_v2): quick visual checks of baseline vs optimized and optimization trajectory.
- optional adapters (`--save-dir`): saved LoRA weights for trained adversary runs.

How to use them:
- compare experiments quickly with metrics JSON + manifest fields.
- debug behavior with row-level CSVs (`prompt`, rewritten prompt, response, reward/verdict).
- inspect prompt optimization quality via trace CSV + plots.

---

## 7) How code is organized (for contributors)

## Top-level layout

- `runs/`: entrypoint orchestration scripts (CLI, run mode, phase order, artifact wiring).
- `src/`: reusable modules used by all runs.
- `scripts/`: wrappers/launchers (`run_unified_experiment.py`, `launch_unified_prime.sh`).
- `configs/default.yaml`: central default configuration.

## Main modularity components

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
  - shared orchestration helpers used across runs (prompt pooling/splitting, reward+verdict normalization, adapter save helper)

## Canonical run-script shape to preserve

When adding/editing run code, keep this order:
1. parse args + defaults
2. resolve device + evaluation config
3. build long-lived sessions
4. load/slice dataset
5. baseline eval
6. optimize/eval loop
7. final eval
8. persist artifacts + manifest

This consistency helps new teammates inspect any run and recognize the same structure quickly.

---

## 8) Suggested teammate workflow for making changes

1. Pick one run script as the target behavior.
2. Keep reusable logic in `src/`, not in `runs/`.
3. Keep run-specific control flow in `runs/`.
4. Update docs when changing CLI, mode names, or artifacts:
   - `README.md`
   - `src/README.md`
   - this file (`getting_started.md`)
5. Smoke-test with a short-budget run before large Prime jobs.

---

## 9) Minimal command cheat sheet

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
