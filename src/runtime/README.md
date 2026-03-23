# `src/runtime` Reference

Runtime layer for shared generation, judging, evaluation, and GEPA optimization primitives used by experiment entrypoints in `runs/`.

For end-to-end onboarding and run commands, see:

- `README.md`
- `docs/getting-started.md`
- `src/README.md`

## Config model

Runtime and model defaults are centralized in `configs/default.yaml`.

- `src/runtime/defaults.py` exposes `load_default_config()`.
- Active run scripts load defaults from YAML for model IDs, reflection endpoint, and other stable values.
- CLI is now focused on experiment controls (mode, dataset sizes, budgets, thresholds, outputs), not backend/model selection.

## Public API highlights

- Construction
  - `RuntimeCatalog.build_target_session(...)` (local HF weights; used by vector steering)
  - `RuntimeCatalog.build_openai_target_session(...)` (HTTP target via same vLLM as reflection)
  - `build_vllm_target_session(defaults)` / `build_local_hf_target_session(...)` / `build_reflection_gateway_for_defaults(...)` in `src/runtime/target_factory.py`
  - `RuntimeCatalog.build_adversary_session(...)`
  - `RuntimeCatalog.build_judge_session(...)`
  - `RuntimeCatalog.build_reflection_gateway(...)`

Default experiment runs load **`build_vllm_target_session`** from YAML: `runtime.models.target_model_name` and `reflection_model_name` should match the vLLM `--served-model-name`. **`runs/vector_steering_baseline.py`** is the only script that loads the target model locally (`runs.vector_steering_baseline.target_inference: local_hf`).
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
  - `runs/gepa_run.py` uses it for baseline and optimized prompt evaluation.
  - `runs/coev_v2_run.py` uses it for attacker+defender evaluation sweeps.
- `run_dual_role_gepa_prompt_optimization(...)` is the stage-boundary optimizer for CoEV v2.
  - Requires a `DualRoleGepaContext` with adversary/target sessions and (for judge mode) a judge session.
  - Consumes GEPA train/val examples built from prompt lists via shared adapters in `src/data.py`.

## Design boundaries

- `src/runtime` owns reusable mechanics and backend adapters.
- `runs/*.py` owns orchestration:
  - stage schedules,
  - script mode handling,
  - run-specific artifact naming and manifests.

## Script invocation reference

Best practice is to use `uv run` from repo root and treat `configs/default.yaml` as the source of truth for stable runtime values.

For fuller run mode details and setup guidance, use `docs/getting-started.md`.

### Direct script calls

```bash
# GEPA prompt optimization
uv run runs/gepa_run.py --show-progress

# CoEV baseline runner
uv run runs/coev_run.py --mode reinforce
uv run runs/coev_run.py --mode gepa
uv run runs/coev_run.py --mode eval

# CoEV v2 (staged REINFORCE + dual-role GEPA)
uv run runs/coev_v2_run.py --mode coev
uv run runs/coev_v2_run.py --mode eval
```

### Unified wrapper (recommended for team usage)

```bash
uv run scripts/run_unified_experiment.py --mode gepa
uv run scripts/run_unified_experiment.py --mode coev
uv run scripts/run_unified_experiment.py --mode coev_v2
```

## Logic check summary (current)

- `runs/gepa_run.py`: target and reflection share the vLLM HTTP endpoint; evaluation and budget knobs stay in CLI.
- `runs/coev_run.py` / `runs/adversary_run.py`: target via vLLM (same URL as `runtime.reflection`); launch scripts must export `REFLECTION_VLLM_BASE_URL` when not using defaults.
- `runs/coev_v2_run.py`: same as GEPA for target + reflection.
- `runs/vector_steering_baseline.py`: local HF target only (steering vectors).

## Notes for cluster execution

- Scripts in `runs/` are expected to run in an environment with project dependencies preinstalled.
- Runtime modules avoid run-script-specific CLI assumptions so orchestration stays in `runs/` and wrappers such as `scripts/run_unified_experiment.py`.
