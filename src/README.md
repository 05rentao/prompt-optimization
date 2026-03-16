# `src/` and `src/runtime/` Module Guide

## Overview

`src/` is the shared library layer used by experiment entrypoints in `runs/`. It is intentionally modular so training/evaluation scripts can stay focused on pipeline control flow.

**Primary consumers:**
- `runs/coev_run.py`
- `runs/coev_v2_run.py`
- `runs/gepa_run.py`

### Modularity Rules

- `runs/` owns **orchestration**: CLI args, phase ordering, logging, artifact naming.
- `src/` owns **reusable implementation**: dataset loading, scoring, runtime wrappers, GEPA helpers.
- `src/runtime/` owns **backend-specific adapters** behind stable interfaces.
- New pipeline logic should compose existing `src` modules before adding new abstractions.

---

## Package Layout

| File | Responsibility |
|---|---|
| `src/data.py` | HarmBench subset loading and split shaping |
| `src/evaluators.py` | Refusal heuristics and judge verdict normalization |
| `src/artifacts.py` | Consistent run manifest writing |
| `src/types.py` | Shared typed payloads (e.g. `RunManifest`) |
| `src/runtime/` | Generation/judge/reflection runtime interfaces and implementations |

---

## `src/runtime/` File-by-File

### `interfaces.py`
Core protocol contracts:
- `GenerationRequest`: normalized single-turn chat generation input.
- `TargetRuntime`: protocol with `generate(...)`.
- `JudgeRuntime`: protocol with `judge(...)`.
- `ReflectionGateway`: protocol with `verify(...)`, `smoke_test(...)`.
- `LoRABridge`: protocol for adapter persistence (`save_adapters(...)`).
- `GenerationSession`: lightweight wrapper exposing `generate(...)` and `judge(...)`.

### `config.py`
Dataclasses for runtime construction used as inputs to `RuntimeCatalog`:
- `LocalHFConfig`
- `UnslothAdversaryConfig`
- `HarmbenchJudgeConfig`
- `OpenAIReflectionConfig`

### `catalog.py`
`RuntimeCatalog` — single composition entrypoint:
- `build_target_session(...)`
- `build_adversary_session(...)`
- `build_judge_session(...)`
- `build_reflection_gateway(...)`

### `local_hf_runtime.py`
`LocalHFChatRuntime` — loads local Transformers model/tokenizer (optional 4-bit), freezes params, and implements `generate(...)` via chat template. Best for frozen target-model inference.

### `unsloth_adversary_runtime.py`
`UnslothAdversaryRuntime` — loads Unsloth model with LoRA, exposes `sample_policy(...)` with token logprob metadata and `save_adapters(...)` for checkpointing. Best for trainable adversary flows.

### `harmbench_judge_runtime.py`
`HarmbenchJudgeRuntime` — loads HarmBench classifier and implements `judge(...)` returning yes/no verdicts. Use for ASR-style scoring.

### `openai_reflection_gateway.py`
`OpenAIReflectionGateway` — OpenAI-compatible client with `verify(...)`, `smoke_test(...)`, and `bind_openai_env()` context manager. Use for frozen reflection model routing in GEPA runs.

### `env.py`
- `resolve_hf_token()`: resolves HF token from `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.
- `scoped_env(...)`: temporary env var override context manager.

### `evaluation.py`
- `EvaluationConfig`, `EvaluationResult`
- `evaluate_outputs(...)` supporting methods:
  - `"heuristic"` — refusal-score based
  - `"judge"` — HarmBench judge verdict based

### `gepa_prompt_optimization.py`
- `GepaPromptOptimizationConfig`
- `GepaRefusalEvaluator` (callable evaluator class)
- `run_gepa_prompt_optimization(...)`

---

## Quick Start
```python
from src.runtime import (
    RuntimeCatalog,
    LocalHFConfig,
    UnslothAdversaryConfig,
    HarmbenchJudgeConfig,
    OpenAIReflectionConfig,
    GenerationRequest,
    resolve_hf_token,
)

target_session = RuntimeCatalog.build_target_session(
    LocalHFConfig(model_id="meta-llama/Llama-2-7b-chat-hf")
)
adversary_session = RuntimeCatalog.build_adversary_session(
    UnslothAdversaryConfig(model_id="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
)
judge_session = RuntimeCatalog.build_judge_session(HarmbenchJudgeConfig())
reflection_gateway = RuntimeCatalog.build_reflection_gateway(
    OpenAIReflectionConfig(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY")
)

text = target_session.generate(
    GenerationRequest(system_prompt="You are safe.", user_prompt="Hello"),
    device="cuda",
)
```

---

## End-to-End Dataflow

### CoEV path (`runs/coev_run.py`)
1. Parse run config and resolve device.
2. Build sessions via `RuntimeCatalog`: adversary (Unsloth+LoRA), target (local HF), judge (HarmBench).
3. Load train/validation prompts via `load_harmbench_subset`.
4. For each training step: adversary rewrites prompt → target generates defended answer → judge computes reward/ASR → optimizer updates adversary policy.
5. Persist CSV logs, optional adapters, and `RunManifest`.

### GEPA path (`runs/gepa_run.py`)
1. Parse run config and resolve device.
2. Load HarmBench train/val data.
3. Build target session, optional judge session, and reflection gateway.
4. Evaluate baseline system prompt via shared evaluation helpers.
5. Run `run_gepa_prompt_optimization(...)` to produce candidate prompts.
6. Evaluate best candidate and save metrics/tables/plots/manifest.

### CoEV v2 path (`runs/coev_v2_run.py`)
1. Parse run config and resolve device.
2. Build adversary/target/judge sessions plus reflection gateway.
3. Run staged REINFORCE updates (same weight-update semantics as CoEV v1).
4. At each stage boundary, run GEPA-based optimization for:
   - attacker instruction (maximize attack success objective),
   - defense prompt (maximize refusal objective).
5. Re-evaluate final evolved prompts and persist:
   - train/stage logs,
   - optimizer traces,
   - baseline-vs-optimized metrics,
   - plots and `RunManifest`.

---

## Implementation Guidance

- Prefer importing from the package root: `from src.runtime import ...`, `from src.data import ...`, `from src.evaluators import ...`
- Keep runtime objects long-lived — construct once per run.
- Keep evaluation method selection centralized through `EvaluationConfig`.
- When adding a backend, implement it in `src/runtime/` and register it in `RuntimeCatalog` rather than branching in run scripts.
- Preserve backward-compatible CLI flags in `runs/` where practical; map old flags to new implementations internally.

### New CoEV v2 usage

```bash
uv run runs/coev_v2_run.py \
  --mode coev \
  --dataset-name walledai/HarmBench \
  --reflection-model-name meta-llama/Llama-3.1-8B-Instruct \
  --reflection-vllm-base-url http://127.0.0.1:8001/v1 \
  --max-metric-calls 80 \
  --stages 2 \
  --iters-per-stage 5 \
  --results-dir results/coev_v2
```

## Extending the System

1. Create a new script in `runs/` for orchestration only.
2. Reuse `load_harmbench_subset`, `evaluate_outputs`, and runtime sessions.
3. Add new shared behavior to `src/` only if at least one existing pipeline can also reuse it.
4. Emit a `RunManifest` via `write_run_manifest` for consistency across runs.

## What Belongs Where

**Keep in `src/runtime/`:** model bootstrapping, generation logic, evaluation aggregation, environment/token handling, GEPA/refusal evaluators.

**Keep in `runs/` scripts:** CoEV stage scheduling and update strategy, script-specific artifact/report structure, pipeline-specific branching semantics and CLI contracts.