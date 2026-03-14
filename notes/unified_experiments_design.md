# Unified Experiments Design: `run_coev.py` + `mark_exp.py`

## Goal
Consolidate the two experiment tracks into one coherent codebase while preserving:
- CoEV-style adversary training behavior from `experimental_code/run_coev.py`.
- GEPA prompt-optimization workflow from `experimental_code_2/mark_exp.py`.
- Practical execution on Prime Intellect with a single H100-80GB GPU in a `uv` project.

## What Is Similar Today
- Both are HarmBench-centered safety experiments and use harmful prompts as core workload.
- Both have a loop structure of:
  - load data,
  - generate candidate behavior/prompt,
  - query a target model,
  - score outcome,
  - iterate optimization.
- Both serialize run artifacts (CSV/metrics/logging) and depend on repeatable CLI execution.
- Both are trying to improve safety behavior, but with different optimization objects.

## Key Differences That Matter

### 1) Optimization target
- `run_coev.py`: optimizes an attacker/rewriter policy (LoRA-trained adversary) using reinforcement-style updates.
- `mark_exp.py`: optimizes a **system prompt candidate** via GEPA (`optimize_anything`) with reflection.

### 2) Runtime architecture
- `run_coev.py`: local model loading via `transformers`/`unsloth` (adversary + target + judge in-process or locally loaded).
- `mark_exp.py`: OpenAI-compatible API clients against one or two local vLLM endpoints (task/reflection split).

### 3) Evaluator semantics
- `run_coev.py`: HarmBench classifier reward (`yes/no`) defines success of attacks (ASR-centric).
- `mark_exp.py`: refusal-heuristic score (`refusal_score`) as optimization metric, with baseline vs optimized comparison.

### 4) Data handling maturity
- `run_coev.py`: direct prompt slice usage from HarmBench, minimal normalization.
- `mark_exp.py`: robust normalization (`PROMPT_KEYS`, `ID_KEYS`), deterministic shuffle/split, dataset preview output.

### 5) Artifact/reporting style
- `run_coev.py`: step-wise training logs (stage/iter/reward/verdict/ASR).
- `mark_exp.py`: richer run package (metrics JSON, baseline/optimized CSVs, optimizer trace, plots, optimized prompt text).

## Shared Pipeline Pieces To Modularize First

### A) Dataset module
Unify HarmBench loading, normalization, split logic, and optional preview/export:
- One canonical sample schema (`id`, `prompt`, optional metadata).
- Deterministic split utility for train/val/eval.
- Pluggable dataset source (HF dataset, local CSV, or cached copy).

### B) Runtime adapter layer
Abstract model invocation behind runtime adapters:
- `LocalRuntimeAdapter`: for direct `transformers`/`unsloth` execution.
- `OpenAICompatRuntimeAdapter`: for vLLM/OpenAI-compatible endpoints.

This enables both experiment families to share the same orchestrator and evaluator code.

### C) Evaluator abstraction
Support interchangeable scoring modules:
- HarmBench judge (classifier-based).
- Refusal heuristic evaluator.
- Optional hybrid/weighted evaluator for combined signal.

### D) Optimization strategy interface
Define optimizer strategies with a shared contract:
- `ReinforceStrategy` (from CoEV training dynamics).
- `GepaPromptStrategy` (from `optimize_anything` flow).
- Optional `HybridCoevGepaStrategy` for staged/cooperative optimization.

### E) Artifact writer
One run-manifest + standardized outputs:
- run metadata (`config`, git SHA if available, seed, runtime profile),
- per-step trace,
- aggregate metrics,
- optional plots.

## Important Design Decisions To Unify Both

### Decision 1: Keep two entrypoints, one shared core
- Keep `coev` and `mark` as separate experiment modes for continuity.
- Move implementation to shared modules so both call the same pipeline primitives.
- Rationale: lowest migration risk and easiest validation against existing behavior.

### Decision 2: Normalize experiment contracts, not experiment identity
- Do **not** force immediate algorithmic merge.
- Standardize interfaces (dataset in, runtime call, score out, artifacts out), then compose.
- Rationale: prevents semantic drift while making future merges easy.

### Decision 3: Separate "objective" from "evaluator"
- Objective controls what is optimized (attacker policy vs system prompt).
- Evaluator controls how outcomes are scored (judge, refusal heuristic, hybrid).
- Rationale: decouples research questions and enables quick ablations.

### Decision 4: Introduce runtime profiles for single-H100
Define explicit runtime profiles:
- `profile=dual_vllm`: two vLLM servers (task + reflection) with conservative memory split.
- `profile=sequential`: run task and reflection phases sequentially to avoid fragmentation/OOM.
- `profile=local_judge_remote_task` (optional): local judge with remote/in-process task runtime.

Rationale: one GPU requires explicit operational modes, not implicit assumptions.

### Decision 5: Standardize artifact schema
- Every run writes `run_manifest.json` with:
  - mode, models, endpoints, budget, dataset slice, seed, runtime profile.
- Keep mode-specific CSVs but add a common summary payload.
- Rationale: enables direct cross-experiment comparisons and plotting.

## Interesting Design Choices To Explore
- Unified metric:
  - combine HarmBench judge score + refusal heuristic with tunable weights.
- Curriculum schedules:
  - early cheap heuristic scoring, later expensive judge scoring.
- Candidate caching:
  - cache `(system_prompt, user_prompt, model, params) -> response` to reduce repeated calls.
- Asynchronous batch evaluation:
  - especially for vLLM endpoint mode to increase throughput.
- Reflection cadence:
  - call reflection every `N` steps, not every step, to reduce latency and VRAM pressure.
- Defense+attack co-evolution:
  - alternate attacker policy updates and defense prompt updates in staged rounds.

## Prime Intellect (Single H100-80GB) Recommendations
- Prefer conservative defaults first:
  - task and reflection GPU utilization sum <= ~0.90 when dual servers are active,
  - leave headroom for CUDA/runtime overhead.
- If instability or OOM appears, switch to sequential profile:
  - start task server only for evaluation pass,
  - stop task server, start reflection server for optimization step.
- Keep smoke-test preset:
  - tiny train/val sizes + low optimization budget for quick health checks.
- Use `uv run ...` for all entrypoints and scripts to avoid environment drift.

## Suggested Near-Term Integration Sequence
1. Write shared config + dataset + evaluator interfaces.
2. Port `mark_exp.py` to shared modules (it already has stronger dataset/reporting structure).
3. Port `run_coev.py` to shared strategy interface while preserving reward semantics.
4. Add unified runner that dispatches `--mode coev|mark|hybrid`.
5. Add profile-based launcher presets for single H100.

## Implementation Status (Current)
- Completed: shared base modules under `src/experiments/` for:
  - dataset loading/normalization (`data.py`),
  - evaluator helpers (`evaluators.py`),
  - typed run metadata (`types.py`),
  - standardized run manifest writer (`artifacts.py`).
- Completed: `experimental_code_2/mark_exp.py` now uses shared dataset/evaluator/artifact modules and writes `results/run_manifest.json`.
- Completed: `experimental_code/run_coev.py` now uses the shared HarmBench loader and writes a standardized run manifest.
- In progress: replacing mode-specific internals with a common strategy/orchestrator layer.
- Completed: unified runner added at `scripts/run_unified_experiment.py` with `--mode mark|coev|hybrid`.
- Completed: Prime launcher preset added at `scripts/launch_unified_prime.sh` with conservative single-H100 defaults.
- Next: replace remaining mode-specific internals with shared optimizer strategy/orchestrator abstractions.

## Unified Run Surface (Now Available)
- Direct unified runner:
  - `uv run python scripts/run_unified_experiment.py --mode mark`
  - `uv run python scripts/run_unified_experiment.py --mode coev --coev-mode reinforce`
  - `uv run python scripts/run_unified_experiment.py --mode hybrid --hybrid-order mark_then_coev`
- Prime launcher preset:
  - `MODE=mark bash scripts/launch_unified_prime.sh`
  - `MODE=coev COEV_MODE=gepa bash scripts/launch_unified_prime.sh`
  - `MODE=hybrid HYBRID_ORDER=mark_then_coev bash scripts/launch_unified_prime.sh`

The launcher keeps default GPU splits conservative (`0.40 + 0.40`) to leave headroom on one H100.

## Non-Goals For Initial Consolidation
- No immediate deletion of legacy entrypoints.
- No forced replacement of one evaluator with another.
- No deep algorithmic rewrite before parity checks.

## Definition of Done (for first consolidation milestone)
- Both experiment modes run through shared modules.
- Both modes produce standardized run manifests and comparable summary metrics.
- Existing launch workflows still function via compatibility wrappers.
- Single-H100 profile docs and defaults are validated with smoke runs.
