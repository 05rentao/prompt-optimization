# `src/runtime` Reference

Runtime layer for shared generation, judging, evaluation, and GEPA optimization primitives used by experiment entrypoints.

## Public API highlights

- Construction
  - `RuntimeCatalog.build_target_session(...)`
  - `RuntimeCatalog.build_adversary_session(...)`
  - `RuntimeCatalog.build_judge_session(...)`
  - `RuntimeCatalog.build_reflection_gateway(...)`
- Evaluation
  - `EvaluationConfig`, `EvaluationResult`
  - `EvaluatedSample`, `EvaluationBatchResult`
  - `evaluate_outputs(...)`
  - `evaluate_examples(...)`
- GEPA optimization
  - `GepaPromptOptimizationConfig`
  - `run_gepa_prompt_optimization(...)`
  - `DualRoleGepaPromptOptimizationConfig`
  - `DualRoleGepaContext`
  - `DualRoleGepaOptimizationResult`
  - `run_dual_role_gepa_prompt_optimization(...)`

## Execution contracts

- `evaluate_examples(...)` centralizes per-example aggregation and metric shaping.
  - `runs/gepa_run.py` uses it for baseline/optimized prompt evaluation.
  - `runs/coev_v2_run.py` uses it for attacker+defender evaluation sweeps.
- `run_dual_role_gepa_prompt_optimization(...)` is the stage-boundary optimizer for CoEV v2.
  - It requires a `DualRoleGepaContext` with adversary/target sessions and (for judge-mode) a judge session.
  - It consumes GEPA train/val examples built from prompt lists via shared data adapters in `src/data.py`.

## Design boundaries

- `src/runtime` owns reusable mechanics and backend adapters.
- `runs/*.py` owns orchestration:
  - stage schedules,
  - CLI contracts,
  - run-specific artifact naming.

## Dual-role GEPA flow (CoEV v2)

`run_dual_role_gepa_prompt_optimization(...)` performs one stage-boundary update:

1. Build stage-local GEPA train/val examples from prompts.
2. Optimize attacker instruction candidate.
3. Optimize defense prompt candidate.
4. Return updated prompts plus traces/results for logging and artifact generation.

This keeps CoEV stage semantics in the run script while centralizing optimizer behavior in runtime.

## Notes for cluster execution

- Scripts in `runs/` are expected to execute on Prime Intellect environments with project dependencies preinstalled.
- Runtime modules intentionally avoid runner-specific CLI assumptions so orchestration remains in `runs/` and wrappers such as `scripts/run_unified_experiment.py`.
