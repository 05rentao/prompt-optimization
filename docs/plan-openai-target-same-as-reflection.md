# Plan: Route target generation through the same vLLM model as GEPA reflection

This document describes a **planned** architectural change: serve **defender / target** completions via the **same OpenAI-compatible vLLM endpoint** already used for **GEPA reflection**, instead of loading a **second** full model with `LocalHFChatRuntime` in the training process.

**Status:** **implemented** in the main codebase: target generation for GEPA/CoEV/adversary runs uses the same OpenAI-compatible vLLM endpoint as reflection (`build_vllm_target_session` in `src/runtime/target_factory.py`). The **exception** is `runs/vector_steering_baseline.py`, which keeps local HF weights (`runs.vector_steering_baseline.target_inference: local_hf`). This file remains useful background on motivations and experiment semantics.

---

## 1. Motivations

### 1.1 GPU memory (primary)

On a **single GPU** (e.g. 80 GB A100), the current CoEV / GEPA stack often loads:

| Consumer | Typical role |
|----------|----------------|
| **vLLM** | Reflection model (`REFLECTION_MODEL`, e.g. Llama-3.1-8B-Instruct) — separate OS process, `gpu_memory_utilization` budget |
| **Python process** | Unsloth adversary (4-bit) + **local HF target** (`LocalHFChatRuntime`, 4-bit) + HarmBench judge (4-bit NF4 or bf16) |

The **target** duplicates a large LM inside the same machine’s memory budget as vLLM’s weights. Serving **target** and **reflection** from **one** vLLM process removes **one** full local model from the PyTorch process and consolidates inference in the server already started for GEPA.

### 1.2 Operational simplicity

- One **served model ID** and one **port** for both “optimizer/reflection” and “victim model” behavior (subject to experiment design — see §6).
- Easier **Prime Intellect / RunPod** single-GPU workflows: less tuning of `REFLECTION_GPU_UTIL` vs local target VRAM.

### 1.3 Throughput tradeoff (not a motivation, but real)

All **target** rollouts and **GEPA** traffic share **one** vLLM instance. Under heavy concurrency, **latency** and **KV cache** pressure may increase; tuning `--max-num-seqs`, `gpu_memory_utilization`, and client timeouts may be required.

---

## 2. Issues encountered today (context for this repo)

These motivated memory-related workarounds and clarify why “unify target + reflection” is attractive.

### 2.1 CUDA OOM during HarmBench judge / eval

- **Symptom:** `torch.OutOfMemoryError` during `HarmbenchJudgeRuntime` forward (`generate`), often when **~79 GiB** of **79.25 GiB** was already allocated across **multiple processes** (vLLM ~32 GiB + training process ~32 GiB + judge ~14 GiB pattern on one card).
- **Root cause:** Stacking **vLLM** + **local adversary** + **local target** + **bf16 HarmBench Mistral** on one GPU exceeds practical headroom without tuning.

### 2.2 Mitigations already applied (related files)

- **`scripts/launch_coev_v2_rloo_prime.sh`:** Lower default `REFLECTION_GPU_UTIL` (e.g. **0.20**) so vLLM reserves less VRAM; export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the Python run to reduce fragmentation OOMs.
- **`src/runtime/harmbench_judge_runtime.py` + `HarmbenchJudgeConfig`:** Default **`load_in_4bit=True`** (NF4) for the judge; env **`JUDGE_LOAD_IN_4BIT=0`** forces legacy bf16 if needed.
- **`src/runtime/catalog.py`:** Env override when building judge session.

### 2.3 Logic bug (CoEV v2 RLOO runner)

- **Former `coev_v2_RLOO_run.py` (now `runs/coev_v2_run.py --adversary-policy rloo`):** `stage_prompts` was passed into `run_dual_role_gepa_prompt_optimization` without being defined → **`NameError`** at stage boundaries. Fixed by collecting `stage_prompts` per stage (mirror `runs/coev_v2_run.py`).

---

## 3. Current architecture (relevant pieces)

### 3.1 Target interface (stable seam)

`src/runtime/interfaces.py` defines:

- **`GenerationRequest`**: `system_prompt`, `user_prompt`, `max_new_tokens`, `temperature`, `top_p`.
- **`TargetRuntime`**: `generate(self, request: GenerationRequest, device: str) -> str`.
- **`GenerationSession`**: thin wrapper calling `runtime.generate(...)`.

**Any** new backend (OpenAI chat, vLLM HTTP) should implement **`TargetRuntime`** so callers stay unchanged.

### 3.2 Local target today

- **`src/runtime/local_hf_runtime.py` — `LocalHFChatRuntime`**: Loads `AutoModelForCausalLM` with optional `BitsAndBytesConfig` (`LocalHFConfig.use_4bit`), `device_map="auto"`.
- **`src/runtime/catalog.py` — `RuntimeCatalog.build_target_session(LocalHFConfig)`** → `GenerationSession(LocalHFChatRuntime(cfg))`.

### 3.3 Reflection today

- **`src/runtime/openai_reflection_gateway.py` — `OpenAIReflectionGateway`**: `OpenAI` client to `OpenAIReflectionConfig.base_url` / `api_key`; `verify()`, `smoke_test()` via `client.chat.completions.create(...)`.
- GEPA code uses this for **reflection** only, not for scoring rollouts that need the “target” persona.

### 3.4 Config sources

- **`configs/default.yaml`** — `runtime.models`:
  - `target_model_name` — documented for **local** HF paths; for HTTP target runs the **served** model id should match `reflection_model_name` (see `build_vllm_target_session` warnings).
  - `reflection_model_name` — used for GEPA / reflection client and as the OpenAI `model` id for **vLLM-backed** target generation.
- **`runtime.reflection`**: `base_url`, `api_key` (launch scripts override with `REFLECTION_VLLM_BASE_URL` / `REFLECTION_VLLM_API_KEY`).
- **`shared_generation`**: unified default prompts and decoding limits merged into each `runs.<name>` by `load_default_config()` (does not change which **weights** are loaded; only text generation defaults).

**If `target_model_name` and `reflection_model_name` differ** in YAML, the HTTP target path still uses the **reflection** model id for the OpenAI chat call — align IDs for consistent “victim” semantics.

---

## 4. Proposed design

### 4.1 New runtime: OpenAI-compatible target

Add something like **`OpenAIChatTargetRuntime`** (name TBD) implementing **`TargetRuntime`**:

- Constructor takes e.g. **`OpenAIReflectionConfig`** (or a slim **`OpenAITargetConfig`**: `base_url`, `api_key`, `model_id`) + default generation kwargs.
- **`generate(request, device)`**:
  - Map `GenerationRequest` → `chat.completions.create(model=..., messages=[{role:system},{role:user}], max_tokens=..., temperature=..., ...)`.
  - **`device`** is ignored for weights (no local GPU tensors); keep signature for protocol compatibility.

Optional: **reuse** the same `OpenAI` client pattern as `OpenAIReflectionGateway` (shared helper or thin subclass) to avoid duplicating URL/key handling.

### 4.2 Catalog factory

- **`RuntimeCatalog.build_openai_target_session(...)`** or extend **`build_target_session`** with a **mode** / **discriminated config**:
  - **Local HF** → existing path.
  - **OpenAI / vLLM** → new runtime.

### 4.3 Runners: how they obtain `target_session`

Runners today call **`load_target_model(TargetModelConfig)`** → **`LocalHFConfig`**. Planned change:

- Add CLI/YAML flag, e.g. **`--target-backend {local_hf,openai}`** or **`target_backend: openai`** in `configs/default.yaml` under the relevant `runs.*` section.
- When **`openai`**: build target from **`REFLECTION_VLLM_BASE_URL`** + **`reflection_model_name`** (or explicit `target_openai_model_name` if you want override without coupling).

**Important:** Ensure **`reflection_gateway.verify(reflection_model_name)`** and any **smoke test** still run **before** heavy loops; optionally add a **target-specific** one-line generation smoke using the new runtime.

---

## 5. File-by-file impact (implementation checklist)

### 5.1 Core runtime (`src/runtime/`)

| Area | Action |
|------|--------|
| **`interfaces.py`** | Usually **no change** if new class implements `TargetRuntime`. |
| **`config.py`** | Add typed config: e.g. `TargetBackend = Literal["local_hf", "openai"]`, optional `OpenAITargetConfig` fields, or extend `LocalHFConfig` / new dataclass for OpenAI target. |
| **`openai_reflection_gateway.py`** | Optionally extract shared **chat completion** helper used by `smoke_test` and new target runtime (avoid copy-paste). |
| **New file** e.g. **`openai_target_runtime.py`** | Implement `OpenAIChatTargetRuntime(TargetRuntime)`. |
| **`catalog.py`** | New builder or branch in `build_target_session`. |
| **`__init__.py`** | Export new config/types if part of public API. |

### 5.2 GEPA / CoEV (`src/runtime/gepa_prompt_optimization.py`)

- **`DualRoleGepaContext`**, **`_target_generate`**, **`GepaRefusalEvaluator`**, **`AttackerInstructionEvaluator`**, **`run_dual_role_gepa_prompt_optimization`**: They only need a **`GenerationSession`** with **`generate()`** — **no structural change** if the session wraps the new runtime.
- **Verify** `device` is passed through; OpenAI target can no-op on `device`.

### 5.3 Runs (`runs/`)

| File | Notes |
|------|--------|
| **`coev_v2_run.py`** | REINFORCE or RLOO (`--adversary-policy`); `target_generate`, `RunContext`, `evaluate_prompts`, `DualRoleGepaContext` wiring; manifest `models.target_model` string should reflect **served** model id when using OpenAI. |
| **`gepa_run.py`** | Same. |
| **`adversary_run.py`** | Uses `load_target_model` + `target_session.generate`. |
| **`coev_run.py`** | Local `TargetModelConfig`; uses `LocalHFConfig` in `load_target_model`. |
| **`vector_steering_baseline.py`** | **Special case:** accesses **`target_session.runtime.model`** and **`tokenizer`** directly for steering hooks — **cannot** switch to OpenAI target without a **different** architecture (or keeping local target for this script only). **Document explicitly:** either exclude this runner from “unified target” or provide a **local-only** code path. |

### 5.4 Scripts (`scripts/`)

| File | Notes |
|------|--------|
| **`launch_coev_v2_rloo_prime.sh`** | After implementation: document that **target** may use **`REFLECTION_VLLM_BASE_URL`**; optional env **`TARGET_BACKEND=openai`**; revisit **`REFLECTION_GPU_UTIL`** defaults once local target is gone (vLLM carries more load). |
| **`launch_gepa_prime.sh`** | Comment today: “local target by default” — update when OpenAI target is optional. |
| **`launch_unified_prime.sh`** | Same. |
| **`launch_vector_steering_prime.sh`** | Likely **unchanged** if steering stays on **local** target model. |

### 5.5 Configs (`configs/`)

- **`default.yaml` / `smoke.yaml`**: Per-run HTTP target behavior is effectively **OpenAI/vLLM** for GEPA, CoEV v2, and adversary runs (`build_vllm_target_session`). Optional future keys could still be added under `runtime` or `runs.*`, e.g.:
  - `target_backend: local_hf | openai` (not required today — steering is the `local_hf` exception via `runs.vector_steering_baseline.target_inference: local_hf`)
  - Optional `target_openai_model_name` (today fallback: `reflection_model_name`)
- Align **`reflection.base_url`** with launcher `REFLECTION_VLLM_BASE_URL` (already env-overridden in runners).
- **`shared_generation`**: canonical prompts and shared `max_tokens` / temperature defaults; merged into `runs.*` at load time (see `docs/getting-started.md`).

### 5.6 Docs

- **`docs/getting-started.md`**: Update “models” story when unified target is available.
- This file: mark sections **implemented** / **deferred** after work lands.

---

## 6. Experiment and compatibility notes

1. **Changing the victim model:** If `target_model_name` was Llama-2-7b-chat and reflection is Llama-3.1-8B-Instruct, **unifying** means the **defender** is no longer Llama-2 unless you **serve Llama-2-7b** in vLLM for both (possible but uncommon). Treat as a **new experimental condition**; update paper / tables / manifests accordingly.

2. **`GenerationRequest` semantics:** Local HF uses `apply_chat_template` in `LocalHFChatRuntime`. OpenAI path uses **messages** API — behavior may differ slightly from Transformers chat templates. Mitigation: document or add a flag to use a specific template string for OpenAI.

3. **Latency:** Network hop to vLLM vs in-process `generate` — usually acceptable on localhost.

4. **`vector_steering_baseline.py`:** Requires **raw `model` / `tokenizer`** — **out of scope** for pure OpenAI target unless you add a separate local model for steering only (defeats VRAM savings for that script).

---

## 7. Suggested implementation order (for a coding agent)

1. Add **`OpenAIChatTargetRuntime`** + **`TargetRuntime`** tests / smoke (manual: one `generate` call).
2. Extend **`RuntimeCatalog`** + **`config`** types.
3. Wire **`coev_v2_run.py`** (RLOO via `--adversary-policy rloo`) and optionally **`gepa_run.py`**; validate full baseline eval + one GEPA stage.
4. Extend to **`coev_v2_run.py`**, **`adversary_run.py`**, **`coev_run.py`** as needed.
5. Update **`configs/*.yaml`**, **`launch_*_prime.sh`**, manifests, and **`docs/getting-started.md`**.
6. Re-tune **`REFLECTION_GPU_UTIL`** once local target is optional (measure vLLM RSS under load).

---

## 8. Quick reference — where `target_session` flows today

- **Runners:** `load_target_model` → `RuntimeCatalog.build_target_session(LocalHFConfig(...))`.
- **CoEV v2 / RLOO:** `target_generate` → `target_session.generate(GenerationRequest(...))`; `evaluate_prompts`; training loop; `DualRoleGepaContext(..., target_session=...)`.
- **GEPA:** `gepa_prompt_optimization.py` — `_target_generate`, `GepaRefusalEvaluator`, dual-role optimizers — all **`target_session.generate`**.

Replacing **only** the implementation behind **`GenerationSession.runtime`** is the minimal integration surface.

---

## 9. Related environment variables (current / planned)

| Variable | Role today |
|----------|----------------|
| `REFLECTION_VLLM_BASE_URL` | OpenAI base URL for reflection client |
| `REFLECTION_VLLM_API_KEY` | API key for same |
| `REFLECTION_GPU_UTIL` | vLLM memory fraction (launch scripts) |
| `JUDGE_LOAD_IN_4BIT` | `0` → bf16 HarmBench judge |
| `PYTORCH_CUDA_ALLOC_CONF` | e.g. `expandable_segments:True` (launcher) |

Planned (examples): `TARGET_BACKEND`, or YAML-only `target_backend`.

---

*Last updated: reflects unified reflection+target vLLM routing; also mentions `shared_generation` for prompt/sampling defaults (see `src/runtime/defaults.py`).*
