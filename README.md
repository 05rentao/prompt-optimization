# STAT 4830 — Prompt Optimization Runs

This repository contains four experiment pipelines under `runs/` that share a common flow:
an adversary rewrites harmful prompts, a target model responds, and evaluation tracks ASR/refusal.
Some runs only train adversary weights, while others optimize attacker/defense prompts with GEPA.

## Documentation map

Use these docs in this order:

1. `README.md` (this file): project overview and quick start.
2. `docs/getting-started.md`: full onboarding, setup, configuration, and runbook.
3. `src/README.md`: contributor guide for `src/` architecture and extension patterns.
4. `src/runtime/README.md`: runtime APIs, contracts, and runtime-specific internals.

Backwards-compatible alias:
- `getting_started.md` points to `docs/getting-started.md`.

## Quick start

```bash
# Install dependencies
uv sync

# Unified runner (recommended)
uv run python scripts/run_unified_experiment.py --mode gepa
uv run python scripts/run_unified_experiment.py --mode coev --coev-mode reinforce
uv run python scripts/run_unified_experiment.py --mode coev_v2 --coev-v2-mode coev
uv run python scripts/run_unified_experiment.py --mode adversary --adversary-mode train
```

Prime/cluster launcher:

```bash
MODE=gepa bash scripts/launch_unified_prime.sh
MODE=coev COEV_MODE=reinforce bash scripts/launch_unified_prime.sh
MODE=coev_v2 COEV_V2_MODE=coev bash scripts/launch_unified_prime.sh
MODE=adversary ADVERSARY_MODE=train bash scripts/launch_unified_prime.sh
```

## Canonical run pipeline

For consistency, run scripts follow the same high-level phase order:

1. `parse_args()` + load defaults from `configs/default.yaml`
2. `resolve_device(...)` + build `EvaluationConfig`
3. build long-lived runtime sessions via `RuntimeCatalog`
4. load data via `load_harmbench_subset(...)` and slice prompts
5. baseline evaluation
6. optimization loop (or eval-only mode)
7. final evaluation
8. save artifacts + `run_manifest.json`

This is the expected shape when inspecting code in `runs/`.

## Run overview

### `runs/adversary_run.py`
- Purpose: adversary-only REINFORCE fine-tuning (no prompt optimization).
- Pipeline shape: baseline -> train loop -> final eval -> artifacts.
- Artifacts: metrics JSON + training CSV + eval CSV + manifest (+ optional LoRA adapter save).

### `runs/coev_run.py`
- Purpose: legacy CoEV runner with `reinforce`, `gepa`, or `eval` mode.
- Pipeline shape: baseline eval first, then mode-specific optimize/eval path, then manifest.
- Artifacts: mode-specific CSV logs + manifest (+ optional LoRA adapter save).

### `runs/coev_v2_run.py`
- Purpose: staged CoEV pipeline combining REINFORCE updates with dual-role GEPA.
- Pipeline shape: baseline eval -> staged training/evolution -> final eval -> artifact bundle.
- Artifacts: metrics JSON, comparison/trace/stage CSVs, plots, manifest (+ optional LoRA adapter save).

### `runs/gepa_run.py`
- Purpose: GEPA-only system prompt optimization for defense behavior.
- Pipeline shape: baseline eval -> GEPA optimization -> final eval -> artifacts.
- Artifacts: metrics JSON, baseline/optimized CSVs, optimizer trace, plots, manifest.

## Artifact generation contract

Across runs, artifacts are generated after baseline/final metrics are available:

1. build metrics payloads (baseline/final stats + run metadata)
2. write structured tables (CSV)
3. write `run_manifest.json` as the run index
4. optionally save adapters (adversary-capable runs)
5. optionally save plots/extra trace files (GEPA-focused runs)

By default, artifacts are written under each run's `--results-dir` (or config default), unless that run's config points specific files elsewhere.

## Environment notes

- HarmBench dataset loading may require Hugging Face authentication (`HF_TOKEN`).
- GEPA-based runs need an OpenAI-compatible reflection endpoint (`runtime.reflection` in config), commonly local vLLM.
- A paid external API key is not required for local vLLM with `api_key: EMPTY`.

For full setup, configuration, and troubleshooting details, see `docs/getting-started.md`.

## License / course

For STAT 4830 use. Adversarial and harmful behavior data are for research and course purposes only.