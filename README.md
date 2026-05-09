# STAT 4830 — Prompt Optimization Runs

This repository contains several experiment pipelines under [runs/](runs/) that share a common flow:
an adversary rewrites harmful prompts, a target model responds, and evaluation tracks ASR/refusal.
Some runs only train adversary weights, while others optimize attacker/defense prompts with GEPA.

## Submission demo

The repository includes a lightweight Colab-ready demo notebook: [demo.ipynb](demo.ipynb) (link: [colab](https://drive.google.com/file/d/1MMwF2ReSl52dHivvRb3DxNe3ZhAbGXGp/view?usp=sharing))

Open the notebook in Colab and run cells from top to bottom. A Colab GPU runtime is recommended for the live LoRA section.

This notebook is a stripped down version of our coevolutionary loop with tiny budgets: dataset prep, prompt-pool slicing, baseline eval, one policy-gradient finetune round for an adversary LoRA model, one dual-role GEPA-style prompt update, and final results.

## Submission paper

The repository include the final report pdf [STAT_4830_Paper.pdf](STAT_4830_Paper.pdf) for submission. The .tex file can also be found at [STAT_4830_Paper.tex](STAT_4830_Paper.tex).


## Documentation map

Use these docs in this order:

1. [README.md](README.md) (this file): project overview and quick start.
2. [docs/getting-started.md](docs/getting-started.md): full onboarding, setup, configuration, and runbook.
3. [docs/run_on_prime_guide.md](docs/run_on_prime_guide.md): end-to-end Prime/H100 setup and launch guide.
4. [src/README.md](src/README.md): contributor guide for [src/](src/) architecture and extension patterns.
5. [src/runtime/README.md](src/runtime/README.md): **authoritative** runtime API reference (modules, `build_vllm_stack`, reflection `verify`, evaluation metrics). **Where to edit** experiment code vs shared plumbing: **Contributing** section below in this file.

Backwards-compatible alias:
- `getting_started.md` points to [docs/getting-started.md](docs/getting-started.md).

## Project file structure

Use this map to quickly find where to work.

Core project code and launch entrypoints:
- [runs/](runs/): experiment entry scripts (`gepa`, `coev_v2`, `coev_v2_rloo` via unified runner, `adversary`; legacy [runs/coev_run.py](runs/coev_run.py) kept for reference).
- [src/](src/): shared library code used by all runs (data, evaluation, artifacts, runtime adapters).
- [scripts/](scripts/): convenience wrappers for unified CLI and Prime/cluster launchers.
- [configs/](configs/): YAML config presets ([configs/default.yaml](configs/default.yaml), [configs/smoke.yaml](configs/smoke.yaml), [configs/smoke_eval.yaml](configs/smoke_eval.yaml)). Shared prompts and sampling defaults live under `shared_generation` and are merged into each `runs.<name>` block when you load config (see [src/runtime/defaults.py](src/runtime/defaults.py)). [configs/prompt_reference.yaml](configs/prompt_reference.yaml) lists legacy prompt strings for reference only (not loaded by code).
- [data/](data/): local input datasets/resources used by runs.

Documentation and project context:
- [docs/](docs/): user-facing guides, especially [Getting Started](docs/getting-started.md) and [Run on Prime](docs/run_on_prime_guide.md).
- [README.md](README.md): high-level orientation and quick command reference.
- [docs/archive/](docs/archive/): historical branch notes and planning material kept for reference, not needed for normal execution.
- [tests/](tests/): small **pytest** suite (no GPU): [tests/test_shared_generation_config.py](tests/test_shared_generation_config.py) exercises `shared_generation` merge in [src/runtime/defaults.py](src/runtime/defaults.py); [tests/test_policy_gradient.py](tests/test_policy_gradient.py) covers [src/runtime/policy_gradient.py](src/runtime/policy_gradient.py). Run from repo root: `uv run python -m pytest tests/ -q`.
- [reports/](reports/): course/report artifacts (for example [reports/report.md](reports/report.md) and [reports/Lightning Talk_ Prompt Steering.pdf](reports/Lightning%20Talk_%20Prompt%20Steering.pdf)).

Run outputs and experiment artifacts:
- [results/](results/): generated structured run outputs (ignored by default; keep curated summaries/figures separately).
- [outputs/](outputs/): generated model outputs, GEPA traces, vectors, and intermediate exports (ignored by default).
- `logs/`: runtime logs (currently mostly empty, may be populated by long runs).

Legacy:
- `legacy_code/`: older notebooks/prototypes kept locally for reference, ignored for final submission.

Project metadata:
- [pyproject.toml](pyproject.toml): Python project/dependency configuration.
- [uv.lock](uv.lock): locked dependency resolution for reproducible environments.

## Quick start

```bash
# Install dependencies
uv sync

# Unified runner (recommended). Sub-modes and paths: configs/default.yaml -> shared_generation + runs.* -> scripts.unified_runner
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

1. `parse_args()` + load defaults via `load_default_config()` ([configs/default.yaml](configs/default.yaml) or `PROMPT_OPT_CONFIG_PATH`), which merges `shared_generation` into each `runs.<name>` before argparse defaults are applied
2. **`patch_run_args_from_config(...)`** on active pipelines (GEPA, CoEV v2, adversary) attaches `runtime_profile`, model ids from YAML, and effective reflection URL/key (`REFLECTION_VLLM_*` env overrides) onto the argparse namespace for manifests (legacy **[runs/coev_run.py](runs/coev_run.py)** wires YAML fields inline)
3. `resolve_device(...)` + build `EvaluationConfig`
4. Build long-lived sessions: **`build_vllm_stack(defaults)`** where target + reflection gateway must stay paired (GEPA, CoEV v2), or **`build_vllm_target_session`** + adversary/judge via **`RuntimeCatalog`**
5. **`OpenAIReflectionGateway.verify(...)`** uses a minimal chat completion on the configured model id (single local vLLM assumption); then optional **`smoke_test`**
6. load data via `load_harmbench_subset(...)` and slice prompts
7. baseline evaluation
8. optimization loop (or eval-only mode)
9. final evaluation
10. save artifacts + `run_manifest.json`

This is the expected shape when inspecting code in [runs/](runs/). API details: **[src/runtime/README.md](src/runtime/README.md)**.

## Contributing: optional logic vs plumbing

Use this when deciding **where** a change belongs. The goal is to keep **experiment behavior** in easy-to-find places and **shared wiring** stable.

### Prefer editing here (behavior, experiments, course deliverables)

- **[runs/](runs/)** — Stage schedules, argparse, training loops, when to call GEPA or policy-gradient steps, artifact paths, manifest fields. **Start here** for new run modes or different eval cadence.
- **[configs/](configs/)** and especially **`shared_generation`** — Prompts, budgets, thresholds, dataset sizes: most changes need **no Python** if keys already map through `load_default_config()`.
- **[src/evaluators.py](src/evaluators.py)**, **[src/run_pipeline.py](src/run_pipeline.py)**, **[src/artifacts.py](src/artifacts.py)** — Refusal heuristics, shared rewrite/reward helpers, CSV/plot/manifest writers used across runs.
- **[src/runtime/gepa_prompt_optimization.py](src/runtime/gepa_prompt_optimization.py)** — GEPA-specific optimization **logic** (how candidates are scored, dual-role wiring) *after* you understand how it composes with `GenerationSession` and `OpenAIReflectionGateway`.

### Treat as plumbing (wiring models, HTTP, env — change only with intent)

These files coordinate **one** vLLM URL, **one** reflection client, and **consistent** manifests across runners. Casual edits here are likely to break multiple pipelines.

- **[src/runtime/sessions.py](src/runtime/sessions.py)** — `build_vllm_target_session`, `build_vllm_stack`, `patch_run_args_from_config`, reflection URL helpers. Touch when **adding a backend** or fixing URL/env bugs, not for one-off experiment tweaks.
- **[src/runtime/openai_http.py](src/runtime/openai_http.py)** — Chat client, `OpenAIReflectionGateway.verify` / `smoke_test`. Run scripts depend on stable behavior at startup.
- **[src/runtime/contracts.py](src/runtime/contracts.py)** and **[src/runtime/__init__.py](src/runtime/__init__.py)** — Public types and re-exports; renaming or reshaping types forces updates in **[runs/](runs/)** and tests.
- **[src/runtime/defaults.py](src/runtime/defaults.py)** — `shared_generation` merge and `load_default_config`; changes affect **every** run’s effective YAML.

**Rule of thumb:** If the change is “what we optimize” or “how we report results,” lean toward **[runs/](runs/)** + **config**. If it is “how HTTP talks to vLLM” or “how env overrides YAML,” it belongs in **[src/runtime/sessions.py](src/runtime/sessions.py)** / **[src/runtime/openai_http.py](src/runtime/openai_http.py)** and should stay coordinated. Full module reference: **[src/runtime/README.md](src/runtime/README.md)**.

## Run overview

### [runs/adversary_run.py](runs/adversary_run.py)
- Purpose: adversary-only policy-gradient fine-tuning (REINFORCE, RLOO, or rejection sampling via `--adversary-policy` / `--rs-min-successes`; no prompt optimization).
- Pipeline shape: baseline → train loop → final eval → artifacts.
- Artifacts: metrics JSON + training CSV + eval CSV + manifest (+ optional LoRA adapter save). Shared update math: [src/runtime/policy_gradient.py](src/runtime/policy_gradient.py).

### [runs/coev_run.py](runs/coev_run.py)
- Purpose: legacy CoEV runner with `reinforce`, `gepa`, or `eval` mode.
- Pipeline shape: baseline eval first, then mode-specific optimize/eval path, then manifest.
- Artifacts: mode-specific CSV logs + manifest (+ optional LoRA adapter save).

### [runs/coev_v2_run.py](runs/coev_v2_run.py)
- Purpose: staged CoEV with REINFORCE or RLOO adversary updates, optional rejection sampling and multi-query rewards, named `--adversary-prompt` presets, and dual-role GEPA (`runs.coev_v2` in YAML).
- Use `--adversary-policy rloo` for the former separate RLOO entrypoint (unified runner: `--mode coev_v2_rloo`).
- Pipeline shape: baseline eval → staged training/evolution → final eval → artifact bundle.
- Artifacts: metrics JSON, comparison/trace/stage CSVs, plots, manifest (+ optional LoRA adapter save); `run_manifest.json` uses `mode` `coev_v2` or `coev_v2_rloo` by policy.

### [runs/gepa_run.py](runs/gepa_run.py)
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

For full setup, configuration, and troubleshooting details, see [docs/getting-started.md](docs/getting-started.md).

## License / course

For STAT 4830 use. Adversarial and harmful behavior data are for research and course purposes only.
