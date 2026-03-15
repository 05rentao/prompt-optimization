# Runtime Module Guide

This directory contains reusable runtime primitives for experiment scripts such as:

- `runs/coev_run.py`
- `runs/gepa_run.py`

The goal is to keep scripts focused on **orchestration** (what to run) while moving reusable runtime/evaluation/optimization mechanics here (how to run).

## Design Intent

- Keep experiment entrypoints simple and readable.
- Avoid duplicating model bootstrapping and generation logic.
- Make GEPA/refusal evaluation reusable across future pipelines.
- Centralize environment/token handling in one place.

## Quick Start

Typical setup pattern in a script:

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

## File-by-File Overview

### `interfaces.py`

Core contracts:

- `GenerationRequest`: normalized single-turn chat generation input.
- `TargetRuntime`: protocol with `generate(...)`.
- `JudgeRuntime`: protocol with `judge(...)`.
- `ReflectionGateway`: protocol with `verify(...)`, `smoke_test(...)`.
- `LoRABridge`: protocol for adapter persistence (`save_adapters(...)`).
- `GenerationSession`: lightweight wrapper around a runtime object; exposes:
  - `generate(...)`
  - `judge(...)`

Use this when you want a consistent call surface from scripts.

### `config.py`

Dataclasses for runtime construction:

- `LocalHFConfig`
- `UnslothAdversaryConfig`
- `HarmbenchJudgeConfig`
- `OpenAIReflectionConfig`

Use these as inputs to `RuntimeCatalog`.

### `catalog.py`

`RuntimeCatalog` is the composition entrypoint:

- `build_target_session(...)`
- `build_adversary_session(...)`
- `build_judge_session(...)`
- `build_reflection_gateway(...)`

Scripts should generally construct runtime objects through this catalog.

### `local_hf_runtime.py`

`LocalHFChatRuntime`:

- Loads local Transformers model/tokenizer (optional 4-bit).
- Freezes model params for inference.
- Implements `generate(...)` using chat template + `model.generate`.

Best for frozen target-model inference in experiments.

### `unsloth_adversary_runtime.py`

`UnslothAdversaryRuntime`:

- Loads Unsloth model and attaches LoRA.
- Exposes `sample_policy(...)` with token logprob metadata.
- Exposes `save_adapters(...)` for adapter checkpointing.

Best for trainable adversary flows (`runs/coev_run.py` style).

### `harmbench_judge_runtime.py`

`HarmbenchJudgeRuntime`:

- Loads HarmBench classifier model/tokenizer.
- Implements `judge(...)` returning yes/no classifier completions.

Use this for ASR-style scoring where HarmBench judge is required.

### `openai_reflection_gateway.py`

`OpenAIReflectionGateway`:

- Creates OpenAI-compatible client.
- `verify(...)` checks endpoint + model availability.
- `smoke_test(...)` does a quick completion sanity check.
- `bind_openai_env()` context manager temporarily routes global OpenAI env vars.

Use this for frozen reflection model routing during GEPA runs.

### `env.py`

Shared environment utilities:

- `resolve_hf_token()`: resolves HF token from `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.
- `scoped_env(...)`: temporary env var override context manager.

Use this instead of re-implementing env logic in scripts.

### `evaluation.py`

Reusable output scoring:

- `EvaluationConfig`
- `EvaluationResult`
- `evaluate_outputs(...)` with methods:
  - `"heuristic"` (refusal-score based)
  - `"judge"` (HarmBench judge verdict based)

Use this when you want consistent metrics across pipelines.

### `gepa_prompt_optimization.py`

Reusable GEPA optimization pieces:

- `GepaPromptOptimizationConfig`
- `GepaRefusalEvaluator` (callable evaluator class)
- `run_gepa_prompt_optimization(...)`

Use this when integrating GEPA in any pipeline without rewriting evaluator/config plumbing.

## Recommended Usage Pattern

1. Keep script-level `main()` as orchestration only:
   - parse args
   - load data
   - construct runtime sessions/gateways
   - call runtime helpers
   - save artifacts
2. Put reusable mechanics into this runtime package.
3. Avoid moving experiment-specific policy logic that is unique to one script.

## What Not To Move Here

Keep script-specific central logic in experiment files, for example:

- CoEV stage scheduling and update strategy.
- Script-specific artifact/report structure.
- Pipeline-specific branching semantics and CLI contracts.

This runtime package should hold reusable building blocks, not own each experiment's full control flow.

