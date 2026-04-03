# `src/runtime` — full reference

Runtime layer for shared generation, judging, evaluation, and GEPA optimization. Experiment entrypoints live in `runs/`; this package supplies **stable adapters** and **YAML/env wiring** so runners stay thin.

**Related docs (read in order for onboarding):**

| Doc | Purpose |
|-----|---------|
| `README.md` (repo root) | Project map, quick start, run overview |
| `docs/getting-started.md` | Setup, config, runbook |
| `src/README.md` | Contributor guide for all of `src/` |

---

## Current layout (nine Python modules)

| Module | Role |
|--------|------|
| `contracts.py` | Dataclasses (`LocalHFConfig`, `OpenAITargetConfig`, `ModelConfig`, …), protocols (`TargetRuntime`, `ReflectionGateway`, …), `GenerationRequest`, `GenerationSession` |
| `defaults.py` | `load_default_config`, `shared_generation` merge, `resolve_hf_token`, `scoped_env`, `build_config_snapshot` (lazy-imports reflection URL helper for manifests) |
| `openai_http.py` | `openai_chat_completion`, `OpenAIChatTargetRuntime`, `OpenAIReflectionGateway` (HTTP vLLM / OpenAI-compatible) |
| `local_runtimes.py` | `LocalHFChatRuntime`, `HarmbenchJudgeRuntime`, `UnslothAdversaryRuntime` |
| `sessions.py` | `RuntimeCatalog`, `build_vllm_target_session`, `build_vllm_stack`, `build_local_hf_target_session`, `build_reflection_gateway_for_defaults`, `patch_run_args_from_config`, `resolve_reflection_env_overrides`, timed target helpers |
| `evaluation.py` | `EvaluationConfig`, `evaluate_outputs`, `evaluate_examples`, aggregates |
| `gepa_prompt_optimization.py` | GEPA and dual-role GEPA |
| `policy_gradient.py` | REINFORCE, RLOO, rejection-sampling batch updates |
| `__init__.py` | Public re-exports |

**Removed paths (do not import):** legacy module names such as `config`, `interfaces`, `catalog`, `env`, `openai_chat`, `openai_target_runtime`, `openai_reflection_gateway`, `local_hf_runtime`, `harmbench_judge_runtime`, `unsloth_adversary_runtime`, `target_factory`, `target_session_factory`, `timed_target` as separate files — functionality lives in `contracts.py`, `sessions.py`, `openai_http.py`, `local_runtimes.py`, `defaults.py` as documented above.

---

## Public import surface

Prefer **`from src.runtime import …`** for anything listed in `__init__.py`, including:

- **Sessions / factories:** `RuntimeCatalog`, `build_vllm_target_session`, `build_vllm_stack`, `build_local_hf_target_session`, `build_reflection_gateway_for_defaults`, `patch_run_args_from_config`, `resolve_reflection_env_overrides`, `timed_target_generate`, `cap_thread_workers`, `run_target_requests_ordered`
- **HTTP types:** `OpenAIReflectionGateway` (re-exported; implementation lives in `openai_http.py`)
- **Contracts:** `GenerationRequest`, `GenerationSession`, `HarmbenchJudgeConfig`, `ModelConfig`, `UnslothAdversaryConfig`, `CoevConfig`, `TargetModelConfig`, …
- **Evaluation:** `EvaluationConfig`, `EvaluationResult`, `EvaluatedSample`, `evaluate_outputs`, `evaluate_examples`
- **GEPA:** `GepaPromptOptimizationConfig`, `run_gepa_prompt_optimization`, dual-role types and `run_dual_role_gepa_prompt_optimization`
- **Defaults:** `load_default_config`, `build_config_snapshot`, `resolve_hf_token`, `scoped_env`, …

Submodules such as `policy_gradient` are imported explicitly when needed: `from src.runtime.policy_gradient import …`.

---

## Configuration and env

- **Loader:** `load_default_config()` reads `PROMPT_OPT_CONFIG_PATH` or `configs/default.yaml`, applies `shared_generation` into each `runs.<name>` block, and returns one merged dict.
- **Reflection HTTP credentials:** `_reflection_urls_from_runtime(runtime)` (internal) applies **`REFLECTION_VLLM_BASE_URL`** / **`REFLECTION_VLLM_API_KEY`** over YAML `runtime.reflection`. **`resolve_reflection_env_overrides(defaults)`** calls that with `defaults["runtime"]`. All vLLM-backed builders (`build_vllm_target_session`, `build_reflection_gateway_for_defaults`, `build_target_session_from_runtime` OpenAI branch) go through this path so env overrides stay consistent.
- **Run scripts** call **`patch_run_args_from_config(defaults, args, run="gepa" | "coev_v2" | "adversary")`** after `parse_args` to attach `runtime_profile`, model ids from `runtime.models`, and reflection URL/key onto `args` for manifests and logging.

---

## vLLM / HTTP target + reflection wiring

- **`build_vllm_target_session(defaults)`** builds an **`OpenAIChatTargetRuntime`** using the **same** base URL and API key as reflection, with OpenAI `model=` **`reflection_model_name`** (must match vLLM `--served-model-name`). If `target_model_name` ≠ `reflection_model_name`, a **warning** is issued; the HTTP path still uses the reflection model id.
- **`build_vllm_stack(defaults)`** returns **`(target_session, reflection_gateway)`** as a pair so GEPA / CoEV v2 runners do not drift between two independent factory calls.
- **`build_target_session_from_runtime(runtime_defaults, …)`** selects local HF vs OpenAI using `resolve_target_backend`; the OpenAI branch uses **`_reflection_urls_from_runtime`** (same as above). For HTTP it uses **`target_model_name`** as the chat `model` id (vector steering / alternate entrypoints — align YAML if you also use `build_vllm_target_session` elsewhere).

---

## `OpenAIReflectionGateway` (`openai_http.py`)

- **`verify(reflection_model_name)`** checks the server with a **minimal chat completion** (`max_tokens=1`) on that model id, with **retries** for the common “TCP open before API ready” case. It does **not** call `GET /v1/models`; the assumed setup is **one local vLLM** serving a single model id that matches config.
- **`smoke_test(reflection_model_name)`** runs a short, readable completion for human-visible sanity checks in run scripts.

---

## Evaluation (`evaluation.py`)

- **`evaluate_outputs`**: heuristic mode fills **`mean_refusal_score`**; judge mode sets it to **`None`** (binary verdicts only).
- **`evaluate_examples`** exposes **`aggregate_score`**: uses **`mean_refusal_score`** when present, otherwise **`refusal_rate`**, so dashboards always get one scalar without faking a soft score under judge mode.
- The **`metrics`** dict includes **`mean_refusal_score`** only when the heuristic path produced it; judge-only runs omit that key instead of writing **`0.0`**.

---

## GEPA (`gepa_prompt_optimization.py`)

Dual-role and single-role GEPA depend on **`GenerationSession.generate`** and **`OpenAIReflectionGateway`**; no separate “target file” beyond **`openai_http`**. Optional user **`logger`** callbacks in **`GepaRefusalEvaluator`** are **not** wrapped in try/except — logger failures propagate.

---

## Policy gradients (`policy_gradient.py`)

Shared REINFORCE / RLOO / rejection-sampling tensor math used by **`runs/coev_v2_run.py`**, **`runs/adversary_run.py`**, and legacy **`runs/coev_run.py`** (inline duplicates in `coev_run` are older).

---

## Execution contracts (runs ↔ runtime)

| Runner | Runtime highlights |
|--------|-------------------|
| `runs/gepa_run.py` | `patch_run_args_from_config(..., run="gepa")`, `build_vllm_stack`, judge session when `eval_method=judge` |
| `runs/coev_v2_run.py` | `patch_run_args_from_config(..., run="coev_v2")`, `build_vllm_stack`, dual-role GEPA |
| `runs/adversary_run.py` | `patch_run_args_from_config(..., run="adversary")`, `build_vllm_target_session` + adversary/judge sessions |
| `runs/coev_run.py` | Legacy; `build_vllm_target_session` directly; prefer `coev_v2_run` for current CoEV + GEPA + shared vLLM |
| `runs/vector_steering_baseline.py` | `build_local_hf_target_session` only (needs weights for steering) |

---

## Design boundaries

- **`src/runtime`**: mechanics, backends, evaluation aggregation, GEPA helpers.
- **`runs/`**: CLI, stage schedules, artifact names, manifests.

**Where to edit experiment logic vs shared plumbing:** see the **Contributing: optional logic vs plumbing** section in the repository root **`README.md`** (not duplicated here).

---

## Script invocation (repo root)

```bash
uv sync
uv run python runs/gepa_run.py
uv run python runs/coev_v2_run.py --mode coev
uv run python scripts/run_unified_experiment.py --mode gepa
```

Use **`uv run`** so the locked environment matches `uv.lock`.

---

## Cluster / Prime notes

- Launch scripts export **`REFLECTION_VLLM_BASE_URL`** (and optionally **`REFLECTION_VLLM_API_KEY`**) before Python starts; `patch_run_args_from_config` and **`build_config_snapshot`** record the effective URL for manifests.
- HarmBench judge: env **`JUDGE_LOAD_IN_4BIT=0`** disables 4-bit judge load (see `RuntimeCatalog.build_judge_session`).

---

*Last updated: reflects consolidated sessions helpers, unified reflection URL resolution, completion-based `verify`, and evaluation metric keys for judge vs heuristic.*
