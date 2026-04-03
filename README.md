# STAT 4830 — Prompt Optimization Runs

This repository contains several experiment pipelines under `runs/` that share a common flow:
an adversary rewrites harmful prompts, a target model responds, and evaluation tracks ASR/refusal.
Some runs only train adversary weights, while others optimize attacker/defense prompts with GEPA.

## Documentation map

Use these docs in this order:

1. `README.md` (this file): project overview and quick start.
2. `docs/getting-started.md`: full onboarding, setup, configuration, and runbook.
3. `docs/run_on_prime_guide.md`: end-to-end Prime/H100 setup and launch guide.
4. `src/README.md`: contributor guide for `src/` architecture and extension patterns.
5. `src/runtime/README.md`: runtime APIs, contracts, and runtime-specific internals.

Backwards-compatible alias:
- `getting_started.md` points to `docs/getting-started.md`.

## Project file structure

Use this map to quickly find where to work.

Core project code and launch entrypoints:
- `runs/`: experiment entry scripts (`gepa`, `coev_v2`, `coev_v2_rloo` via unified runner, `adversary`; legacy `coev_run.py` kept for reference).
- `src/`: shared library code used by all runs (data, evaluation, artifacts, runtime adapters).
- `scripts/`: convenience wrappers for unified CLI and Prime/cluster launchers.
- `configs/`: YAML config presets (`default.yaml`, `smoke.yaml`, `smoke_eval.yaml`). Shared prompts and sampling defaults live under `shared_generation` and are merged into each `runs.<name>` block when you load config (see `src/runtime/defaults.py`). `configs/prompt_reference.yaml` lists legacy prompt strings for reference only (not loaded by code).
- `data/`: local input datasets/resources used by runs.

Documentation and project context:
- `docs/`: user-facing guides, especially [Getting Started](docs/getting-started.md) and [Run on Prime](docs/run_on_prime_guide.md).
- `README.md`: high-level orientation and quick command reference.
- `notes/`: working notes, design docs, and planning material (non-critical for execution). See `notes/coev_v2_future_refactor.md` for policy-gradient modularization notes.
- `tests/`: optional unit tests (e.g. `tests/test_policy_gradient.py` for `src/runtime/policy_gradient.py`).
- `reports/`: course/report artifacts (for example `report.md` and presentation PDF).

Run outputs and experiment artifacts:
- `results/`: structured run outputs (manifests, metrics, traces), including smoke runs.
- `outputs/`: generated model outputs, GEPA traces, vectors, and intermediate exports.
- `logs/`: runtime logs (currently mostly empty, may be populated by long runs).

Legacy:
- `legacy_code/`: older notebooks/prototypes kept for reference during migration.

Project metadata:
- `pyproject.toml`: Python project/dependency configuration.
- `uv.lock`: locked dependency resolution for reproducible environments.

## Quick start

```bash
# Install dependencies
uv sync

# Unified runner (recommended). Sub-modes and paths: configs/default.yaml → shared_generation + runs.* → scripts.unified_runner
uv run python scripts/run_unified_experiment.py --mode gepa
uv run python scripts/run_unified_experiment.py --mode coev_v2
uv run python scripts/run_unified_experiment.py --mode coev_v2_rloo
uv run python scripts/run_unified_experiment.py --mode adversary
```

Prime/cluster launcher:

```bash
MODE=gepa bash scripts/launch_unified_prime.sh
MODE=coev_v2 bash scripts/launch_unified_prime.sh
MODE=coev_v2_rloo bash scripts/launch_unified_prime.sh
MODE=adversary bash scripts/launch_unified_prime.sh
```

## Canonical run pipeline

For consistency, run scripts follow the same high-level phase order:

1. `parse_args()` + load defaults via `load_default_config()` (`configs/default.yaml` or `PROMPT_OPT_CONFIG_PATH`), which merges `shared_generation` into each `runs.<name>` before argparse defaults are applied
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
- Purpose: adversary-only policy-gradient fine-tuning (REINFORCE, RLOO, or rejection sampling via `--adversary-policy` / `--rs-min-successes`; no prompt optimization).
- Pipeline shape: baseline → train loop → final eval → artifacts.
- Artifacts: metrics JSON + training CSV + eval CSV + manifest (+ optional LoRA adapter save). Shared update math: `src/runtime/policy_gradient.py`.

### `runs/coev_run.py`
- Purpose: legacy CoEV runner with `reinforce`, `gepa`, or `eval` mode.
- Pipeline shape: baseline eval first, then mode-specific optimize/eval path, then manifest.
- Artifacts: mode-specific CSV logs + manifest (+ optional LoRA adapter save).

### `runs/coev_v2_run.py`
- Purpose: staged CoEV with REINFORCE or RLOO adversary updates, optional rejection sampling and multi-query rewards, named `--adversary-prompt` presets, and dual-role GEPA (`runs.coev_v2` in YAML).
- Use `--adversary-policy rloo` for the former separate RLOO entrypoint (unified runner: `--mode coev_v2_rloo`).
- Pipeline shape: baseline eval → staged training/evolution → final eval → artifact bundle.
- Artifacts: metrics JSON, comparison/trace/stage CSVs, plots, manifest (+ optional LoRA adapter save); `run_manifest.json` uses `mode` `coev_v2` or `coev_v2_rloo` by policy.

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