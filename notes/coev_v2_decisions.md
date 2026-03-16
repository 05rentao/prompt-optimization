# CoEV v2 Design Decisions

This note explains what changed between `runs/coev_v2_run.py` and the original `runs/coev_run.py`, and why those changes were made.

## High-level difference

- `coev_run.py` performs prompt evolution with a handcrafted function (`gepa_evolve_prompts`) that directly prompts the adversary model to synthesize next-stage attacker/defender prompts.
- `coev_v2_run.py` preserves the same staged REINFORCE training cadence, but replaces that handcrafted prompt-evolution step with GEPA optimizer-driven evolution, aligned with the optimization pattern used by `src/runtime/gepa_prompt_optimization.py`.

In short:
- **same**: stage loop, REINFORCE weight updates, eval cadence.
- **different**: how stage-to-stage prompts are evolved (optimizer-based instead of handcrafted generation).

## Why new classes were introduced

## `GepaOptimizationConfig`

Purpose:
- Encapsulate GEPA-specific optimization budget and runtime controls (`max_metric_calls`, optimization tokens, reflection model).

Why:
- In `coev_run.py`, GEPA-like evolution is hardcoded and tied to local generation settings.
- CoEV v2 needs explicit optimizer knobs to make GEPA behavior tunable without editing logic.
- Separates optimizer settings from core CoEV training settings (`CoevConfig`).

## `AttackerInstructionEvaluator`

Purpose:
- Implements GEPA evaluator contract for attacker candidate prompts.
- Scores candidates by attack success objective:
  - heuristic mode: `1 - refusal_score`,
  - judge mode: binary success from HarmBench verdict.

Why:
- GEPA optimization expects a callable evaluator `(candidate, example) -> (score, side_info)`.
- Attacker and defender objectives are opposite directions, so they should not share the same scorer.
- Keeps attacker objective logic isolated, testable, and traceable.

## `DefensePromptEvaluator`

Purpose:
- Implements GEPA evaluator contract for defense prompt candidates.
- Scores candidates by defense objective:
  - heuristic mode: `refusal_score`,
  - judge mode: non-success from HarmBench verdict.

Why:
- Defense optimization target is not the same as attacker optimization target.
- Explicit evaluator class makes objective direction obvious and avoids accidental metric inversion bugs.
- Enables clean optimizer traces by role (`attacker` vs `defender`).

## Other new functions and why they were added

## `evaluate_prompts(...)`

What it does:
- Runs full adversary rewrite + target generation over a dataset and returns:
  - aggregate metrics (`asr`, `refusal_rate`, `aggregate_score`, latency),
  - row-level dataframe.

Why:
- `coev_run.py` had `evaluate_asr(...)` that returned only compact metrics.
- CoEV v2 needs richer outputs for GEPA-style artifact generation (CSV comparison and per-example exports), similar to `gepa_run.py`.

## `_to_gepa_dataset(...)`

What it does:
- Converts stage prompt lists into GEPA train/validation example shape.

Why:
- CoEV stage logs/prompts are not in the exact GEPA dataset format.
- Adds a consistent adapter layer so GEPA can run at each stage boundary.

## `evolve_prompts_with_gepa(...)`

What it does:
- Replaces handcrafted prompt evolution with two optimizer runs per stage:
  - attacker candidate optimization,
  - defense candidate optimization.
- Returns updated prompts plus trace/result metadata.

Why:
- This is the core requirement of CoEV v2.
- Keeps stage-level coevolution semantics unchanged while swapping the evolution mechanism.

## `save_artifacts(...)`

What it does:
- Persists GEPA-style artifacts for CoEV v2:
  - optimized prompts json,
  - metrics json,
  - baseline vs optimized CSV,
  - per-example eval outputs,
  - optimizer traces,
  - trajectory/comparison plots,
  - run manifest.

Why:
- `coev_run.py` mainly writes training CSV + manifest.
- CoEV v2 requirement asked for artifact generation style similar to `gepa_run.py`.

## CLI and orchestration changes

- Added CoEV v2-specific GEPA flags:
  - reflection endpoint/model flags,
  - GEPA metric-call budget,
  - GEPA evaluator token/temperature controls.
- Added mode wiring in `scripts/run_unified_experiment.py` so `coev_v2` can be launched from unified CLI.

Why:
- CoEV v2 needs both REINFORCE controls and optimizer controls.
- Unified runner integration keeps launch patterns consistent with existing workflows.

## What stayed intentionally the same

- REINFORCE update function and inner-loop weight updates.
- Stage structure (`stages`, `iters_per_stage`, periodic eval).
- Runtime/session composition through `RuntimeCatalog`.
- HarmBench data loading and judge integration.

This keeps CoEV v2 comparable to original CoEV while changing only the prompt evolution mechanism and artifact richness.
