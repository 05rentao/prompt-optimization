# `src/` and `src/runtime/` Guide

This file is the contributor guide for the shared library layer under `src/`.

For onboarding and run commands, use:

- `README.md` (project overview + quick start)
- `docs/getting-started.md` (full setup and runbook)
- `src/runtime/README.md` (runtime API reference); root **`README.md`** has **Contributing: optional logic vs plumbing**

## Overview

`src/` is the shared library layer used by experiment entrypoints in `runs/`. It is intentionally modular so run scripts can stay focused on orchestration and experiment flow.

**Primary consumers**
- `runs/coev_run.py`
- `runs/coev_v2_run.py`
- `runs/gepa_run.py`
- `runs/adversary_run.py`

### Modularity rules

- `runs/` owns orchestration: CLI args, phase ordering, logging, artifact naming.
- `src/` owns reusable implementation: data loading, scoring, artifact helpers, runtime wrappers, GEPA helpers.
- `src/runtime/` owns backend-specific adapters behind stable interfaces.
- New pipeline logic should compose existing `src` modules before adding new abstractions.

## Teammate Quickstart

Use this when you want to stand up or modify a run script quickly.

### 1) Pick a base script

- Start from `runs/gepa_run.py` if you are optimizing prompts with reflection loops.
- Start from `runs/coev_run.py` for baseline CoEV REINFORCE flow.
- Start from `runs/coev_v2_run.py` for staged CoEV + GEPA prompt evolution.
- Start from `runs/adversary_run.py` for adversary-only weight fine-tuning (no prompt optimization).

### 2) Keep the canonical script shape

Most scripts should follow this order:

1. `parse_args()` with all CLI flags; `load_default_config()` for merged YAML.
2. **`patch_run_args_from_config(...)`** (GEPA / CoEV v2 / adversary) where applicable.
3. `resolve_device(...)`.
4. Build long-lived runtime sessions: **`build_vllm_stack`** or **`RuntimeCatalog`** + **`build_vllm_target_session`** as appropriate (see `src/runtime/README.md`).
5. Load dataset via `load_harmbench_subset(...)`.
6. Execute train/eval loop.
7. Write artifacts and `RunManifest`.

### 3) Reuse shared building blocks

- Data: `load_harmbench_subset(...)`.
- Evaluation: `EvaluationConfig`, `evaluate_outputs(...)`, `evaluate_examples(...)`.
- Runtime: `RuntimeCatalog` builders + `GenerationRequest`.
- Artifacts: `write_json(...)`, `write_many_csv(...)`, plotting helpers, `write_run_manifest(...)`.

### 4) Run existing scripts locally

```bash
uv run python runs/gepa_run.py --help
uv run python runs/coev_run.py --help
uv run python runs/coev_v2_run.py --help
uv run python runs/adversary_run.py --help
```

### 5) Best way to run scripts

Use `configs/default.yaml` as the source of truth (model IDs, reflection endpoint, experiment budgets, and `scripts.unified_runner` for the unified wrapper). The unified CLI is only `--mode`.

```bash
# Recommended: use the unified wrapper (details in docs/getting-started.md).
uv run python scripts/run_unified_experiment.py --mode gepa
uv run python scripts/run_unified_experiment.py --mode coev_v2
uv run python scripts/run_unified_experiment.py --mode coev_v2_rloo

# Direct script runs (when developing one pipeline):
uv run python runs/gepa_run.py
uv run python runs/coev_run.py --mode reinforce
uv run python runs/coev_v2_run.py --mode coev
uv run python runs/coev_v2_run.py --mode coev --adversary-policy rloo
uv run python runs/adversary_run.py --mode train
```

### 6) Minimal runtime wiring example

```python
from src.runtime import (
    RuntimeCatalog,
    UnslothAdversaryConfig,
    HarmbenchJudgeConfig,
    OpenAIReflectionConfig,
    GenerationRequest,
    build_vllm_target_session,
    load_default_config,
)

defaults = load_default_config()
target_session = build_vllm_target_session(defaults)
adversary_session = RuntimeCatalog.build_adversary_session(
    UnslothAdversaryConfig(model_id="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
)
judge_session = RuntimeCatalog.build_judge_session(HarmbenchJudgeConfig())
reflection_gateway = RuntimeCatalog.build_reflection_gateway(
    OpenAIReflectionConfig(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY")
)

text = target_session.generate(
    GenerationRequest(system_prompt="You are a safe assistant.", user_prompt="Hello"),
    device="cuda",
)
```

## Guide: Writing a New Run Script

Use this checklist when creating `runs/<your_script>.py`.

### Script checklist

1. Add `#!/usr/bin/env python3` and a top-level module docstring describing the pipeline.
2. Use `argparse` and expose defaults similar to existing scripts.
3. Build configs/sessions once near startup (do not rebuild per example).
4. Keep core loop logic in small testable functions.
5. Centralize metric logic through `EvaluationConfig`.
6. Save structured artifacts (CSV/JSON/plots) and a final `run_manifest.json`.

### Starter template (recommended)

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.artifacts import write_run_manifest
from src.data import load_harmbench_subset
from src.runtime import (
    EvaluationConfig,
    HarmbenchJudgeConfig,
    RuntimeCatalog,
    build_vllm_target_session,
    load_default_config,
    resolve_hf_token,
)
from src.types import RunManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe your run")
    parser.add_argument("--dataset-name", default="walledai/HarmBench")
    parser.add_argument("--dataset-config", default="standard")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results/my_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = resolve_hf_token()

    defaults = load_default_config()
    target_session = build_vllm_target_session(defaults)
    judge_session = RuntimeCatalog.build_judge_session(HarmbenchJudgeConfig())
    eval_cfg = EvaluationConfig(method="judge")

    train_data, val_data, _ = load_harmbench_subset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        hf_token=token,
    )

    # TODO: your train/eval logic here, reusing evaluate_outputs/evaluate_examples.

    manifest = RunManifest(
        script_name="my_run.py",
        mode="custom",
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        train_size=len(train_data),
        val_size=len(val_data),
        seed=args.seed,
        target_model=target_cfg.model_id,
        runtime_profile="local_transformers",
        eval_method=eval_cfg.method,
        results_dir=str(Path(args.results_dir)),
        metrics={},
        notes=[],
    )
    write_run_manifest(Path(args.results_dir), manifest)


if __name__ == "__main__":
    main()
```

### Common pitfalls to avoid

- Building runtime sessions inside loops (slow and error-prone).
- Adding model-specific branching in `runs/` instead of `src/runtime/sessions.py` (`RuntimeCatalog`).
- Reimplementing evaluation math instead of using shared evaluators.
- Skipping `RunManifest` output (breaks downstream analysis consistency).

## Package Layout

| File | Responsibility |
|---|---|
| `src/data.py` | HarmBench subset loading and split shaping |
| `src/evaluators.py` | Refusal heuristics and judge verdict normalization |
| `src/artifacts.py` | Shared artifact writers (json/csv/plots) and run manifest writing |
| `src/types.py` | Shared typed payloads (for example `RunManifest`, eval/train row schemas) |
| `src/runtime/` | Generation/judge/reflection runtime interfaces and implementations |

## `src/runtime/` File-by-File

### `contracts.py`
Runtime dataclasses and protocol contracts:
- Config: `LocalHFConfig`, `TargetModelConfig`, `UnslothAdversaryConfig`, `HarmbenchJudgeConfig`, `OpenAIReflectionConfig`, `OpenAITargetConfig`, and related CoEV/GEPA config types
- Protocols: `GenerationRequest`, `TargetRuntime`, `JudgeRuntime`, `ReflectionGateway`, `LoRABridge`, `GenerationSession`

### `sessions.py`
Runtime composition and helpers:
- `RuntimeCatalog`: `build_target_session`, `build_openai_target_session`, `build_adversary_session`, `build_judge_session`, `build_reflection_gateway`
- YAML/env factories: `build_vllm_target_session`, `build_local_hf_target_session`, `build_reflection_gateway_for_defaults`, `resolve_reflection_env_overrides`, `resolve_target_backend`, `build_target_session_from_runtime`
- Timed target batching: `timed_target_generate`, `cap_thread_workers`, `run_target_requests_ordered`

### Other runtime modules

- `openai_http.py`: OpenAI-compatible HTTP target (`OpenAIChatTargetRuntime`), reflection gateway (`OpenAIReflectionGateway`), and shared `openai_chat_completion` helper.
- `local_runtimes.py`: local Transformers target (`LocalHFChatRuntime`), HarmBench judge (`HarmbenchJudgeRuntime`), PEFT adversary (`UnslothAdversaryRuntime`).
- `defaults.py`: YAML loading, `shared_generation` merge, `resolve_hf_token()`, `scoped_env(...)`.
- `evaluation.py`: `EvaluationConfig`, `EvaluationResult`, `EvaluatedSample`, `EvaluationBatchResult`, `evaluate_outputs(...)`, `evaluate_examples(...)`.
- `gepa_prompt_optimization.py`: GEPA optimization configs and run helpers (single-role and dual-role).

## End-to-End Dataflow

### CoEV path (`runs/coev_run.py`)
1. Parse run config and resolve device.
2. Build adversary/target/judge sessions via `RuntimeCatalog`.
3. Load HarmBench train/val data.
4. Run REINFORCE updates from judge-backed rewards.
5. Save logs and `RunManifest`.

### GEPA path (`runs/gepa_run.py`)
1. Parse run config and resolve device.
2. Build target session, optional judge session, and reflection gateway.
3. Load HarmBench train/val data.
4. Evaluate baseline prompt using shared evaluators.
5. Run `run_gepa_prompt_optimization(...)`.
6. Evaluate optimized prompt and persist artifacts/manifest.

### CoEV v2 path (`runs/coev_v2_run.py`)
1. Parse run config and resolve device.
2. Build adversary/target/judge sessions plus reflection gateway.
3. Run staged REINFORCE or RLOO updates (optional rejection sampling, multi-query rewards).
4. Run dual-role GEPA optimization at stage boundaries.
5. Re-evaluate and save full artifacts + `RunManifest`.

### Adversary-only path (`runs/adversary_run.py`)
1. Parse run config and resolve device.
2. Build adversary/target/judge sessions via `RuntimeCatalog`.
3. Load HarmBench train/val data.
4. Fine-tune adversary weights with policy-gradient (`src/runtime/policy_gradient.py`: REINFORCE, RLOO, or rejection sampling).
5. Evaluate on held-out prompts and save artifacts + `RunManifest`.

## Implementation Guidance

- Prefer importing from package roots (for example `from src.runtime import ...`).
- Keep runtime objects long-lived and construct once per run.
- Keep evaluation mode selection centralized through `EvaluationConfig`.
- When adding a backend, implement it in `src/runtime/` and register in `RuntimeCatalog`.
- Preserve CLI compatibility where practical; map legacy flags internally.

### Example CoEV v2 invocation

```bash
uv run python runs/coev_v2_run.py \
  --mode coev \
  --dataset-name walledai/HarmBench \
  --max-metric-calls 80 \
  --stages 2 \
  --iters-per-stage 5 \
  --results-dir results/coev_v2
```

## What Belongs Where

**Keep in `src/runtime/`**
- model bootstrapping
- generation and judging logic
- evaluation aggregation
- environment/token handling
- GEPA/refusal evaluators

**Keep in `runs/`**
- stage scheduling and update strategy
- pipeline-specific control flow
- script-specific artifact/report structure
- CLI contracts