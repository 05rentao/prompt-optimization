# Experimental Pipeline README

This README documents the **current integrated experimental pipeline** that combines:
- `experimental_code/run_coev.py` (CoEV-style adversary training),
- `experimental_code_2/mark_exp.py` (GEPA system-prompt optimization),
- shared utilities under `src/experiments/`,
- unified orchestration via `scripts/run_unified_experiment.py`,
- Prime launcher preset `scripts/launch_unified_prime.sh`.

It is aligned with the design decisions in `notes/unified_experiments_design.md`.

---

## What Is Integrated Right Now

### Shared core modules
The following shared modules now back both experiment tracks:

- `src/experiments/data.py`
  - HarmBench loading/normalization/splitting (`id`, `prompt` schema)
- `src/experiments/evaluators.py`
  - refusal scoring helpers and metric summarization
- `src/experiments/types.py`
  - typed run metadata (`RunManifest`)
- `src/experiments/artifacts.py`
  - standard `run_manifest.json` writer

### Entry points still preserved
- `experimental_code/run_coev.py`
  - still runs `reinforce|gepa|eval` modes
  - now uses shared HarmBench loading
  - now writes standardized run manifest
- `experimental_code_2/mark_exp.py`
  - still runs baseline -> GEPA optimize -> re-eval
  - now uses shared dataset/evaluator/artifact helpers
  - now writes standardized run manifest

### Unified orchestration layer
- `scripts/run_unified_experiment.py`
  - single CLI with `--mode mark|coev|hybrid`
  - `hybrid` runs both sequentially
  - passes shared dataset/runtime knobs across both paths

---

## Current Pipeline Modes

### 1) `mark` mode (prompt optimization)
1. Load HarmBench train/val subset.
2. Evaluate baseline system prompt on val set.
3. Optimize system prompt with GEPA (`optimize_anything`) using reflection model.
4. Re-evaluate optimized prompt.
5. Save artifacts (CSV/JSON/plots + `run_manifest.json`).

### 2) `coev` mode (adversary optimization)
1. Load HarmBench prompt pool.
2. Run selected mode:
   - `reinforce`: adversary policy updates with reward signal,
   - `gepa`: staged co-evolution style loop,
   - `eval`: evaluation only.
3. Save mode outputs and `run_manifest.json`.

### 3) `hybrid` mode
- Sequentially run both pathways through unified runner:
  - `mark_then_coev` (default), or
  - `coev_then_mark`.

---

## Unified Commands (`uv`)

### Direct unified runner
```bash
# Mark pipeline only
uv run python scripts/run_unified_experiment.py --mode mark

# CoEV pipeline only
uv run python scripts/run_unified_experiment.py --mode coev --coev-mode reinforce

# Sequential hybrid run
uv run python scripts/run_unified_experiment.py --mode hybrid --hybrid-order mark_then_coev
```

### Useful unified flags
```bash
--dataset-name walledai/HarmBench
--dataset-config standard
--dataset-split train
--train-size 100
--val-size 100
--seed 42
--runtime-profile dual_vllm
```

---

## Prime Intellect Single-H100 Launcher

Use:
```bash
bash scripts/launch_unified_prime.sh
```

Default behavior:
- `MODE=mark`
- starts two vLLM endpoints (task + reflection)
- conservative GPU split:
  - `TASK_GPU_UTIL=0.40`
  - `REFLECTION_GPU_UTIL=0.40`

Examples:
```bash
# Mark only
MODE=mark bash scripts/launch_unified_prime.sh

# CoEV only (no vLLM servers launched by script)
MODE=coev COEV_MODE=gepa bash scripts/launch_unified_prime.sh

# Hybrid sequence
MODE=hybrid HYBRID_ORDER=mark_then_coev bash scripts/launch_unified_prime.sh
```

Environment:
- Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) for gated models/datasets.

---

## Artifacts and Output Layout

### Mark path
- Existing outputs are retained (metrics JSON, eval CSVs, plots, optimized prompt).
- Standardized manifest:
  - `<mark_results_dir>/run_manifest.json`

### CoEV path
- Existing CSV outputs are retained.
- Standardized manifest:
  - `<coev_results_dir>/run_manifest.json`

### Unified runner defaults
- Mark results: `results/mark`
- CoEV results: `results/coev`

---

## Runtime Profiles

Current accepted profile string is passed through as metadata/config:
- `dual_vllm` (default for mark/hybrid launch path)
- `local_transformers` (typical coev local execution)
- `sequential` (recommended fallback strategy when VRAM is tight)

Profile-driven behavior is partially implemented now (orchestration + metadata) and will be deepened in future strategy refactors.

---

## Current Limitations

- Optimizer internals are still mode-specific; shared strategy classes are the next step.
- `hybrid` currently means sequential composition of existing scripts, not a single joint optimization loop.
- Some legacy docs (`README.md`) still describe older training flow and may not fully reflect this integrated experimental surface.

---

## Recommended Next Steps

1. Extract shared optimizer strategy interfaces (`ReinforceStrategy`, `GepaPromptStrategy`).
2. Convert `run_coev.py` and `mark_exp.py` into thinner wrappers over a shared orchestrator.
3. Add smoke test commands for all three unified modes.
4. Keep this README in sync with `notes/unified_experiments_design.md` after each integration milestone.
