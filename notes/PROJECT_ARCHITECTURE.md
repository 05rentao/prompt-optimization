# LoRA + GEPA Project Architecture

This document defines the production architecture implied by `legacy_code/Copy_of_basic_implementation_without_gepa (4).ipynb`, and standardizes it into swap-friendly modules for "vibe" experimentation.

## 1) System Overview

The notebook implements a dual-loop optimization pipeline:

- **Inner loop (LoRA / policy learning):** train an adversarial rewriter policy with REINFORCE.
- **Outer loop (GEPA / prompt evolution):** evolve attacker and defender instructions from stage-level outcomes.
- **Evaluation/Judge loop:** score target model behavior via HarmBench-style yes/no classification.

Canonical flow:

1. Sample a harmful/edge-case seed prompt from dataset.
2. Generate adversarial rewrite from the LoRA policy.
3. Feed rewrite to frozen target model (with optional evolving defense prompt).
4. Judge target output as success/failure.
5. Convert verdict to reward.
6. Apply REINFORCE update to LoRA policy.
7. Periodically evolve attacker/defender prompts with GEPA using stage logs.

---

## 2) Core Components

## `AdversaryManager` (LoRA policy orchestration)

**Responsibility**
- Owns policy model (`Qwen` + LoRA adapters), tokenizer, sampling, and policy updates.
- Produces adversarial rewrites and all tensors needed for policy-gradient updates.

**Notebook mapping**
- `policy_sample(...)`
- `adversary_output(...)`
- `reinforce_update_batch_sgd(...)`

**Reference class interface**

```python
class AdversaryManager:
    def sample(self, messages: list[dict], max_new_tokens: int = 120) -> dict: ...
    def update(self, gen_ids: torch.LongTensor, prompt_lens: list[int], rewards: torch.FloatTensor) -> dict: ...
```

**Expected behaviors**
- `sample(...)` applies chat template, generates completion, and returns:
  - full token sequence (`gen_ids`) including prompt + completion
  - `prompt_len` for completion boundary
  - decoded completion text
  - optional logprob diagnostics
- `update(...)` computes token-level logprobs over completion span and applies:
  - `loss = -(rewards * logprob_sums).mean()`
  - optimizer step over LoRA parameters

---

## `BaseOptimizer` (GEPA meta-optimizer)

**Responsibility**
- Evolves prompt-level strategies (attacker instruction and defender system prompt) using stage outcomes.
- Performs meta-analysis over successful/failed attempts and generates next-stage instructions.

**Notebook mapping**
- `gepa_evolve_prompts(stage_logs)`
- `sanitize_gepa_output(text)`

**Reference class interface**

```python
class BaseOptimizer:
    def evolve(self, stage_logs: list[dict]) -> tuple[str, str]: ...
    def sanitize(self, text: str) -> str: ...
```

**Expected behaviors**
- Aggregate stage logs into success/failure sets.
- Construct strict meta-prompts for:
  - attacker strategy evolution
  - defender strategy evolution
- Generate one-line outputs and sanitize filler/meta-tags.
- Return:
  - `next_attacker_instruction: str`
  - `next_defense_prompt: str`

---

## Supporting modules

- **TargetModelGateway** (frozen victim model)
  - notebook: `target_generate(...)` (+ defended variant in loop intent)
- **JudgeGateway**
  - notebook: `generateHarmbenchResults(behavior, generations)`
- **TrainingOrchestrator**
  - notebook training loops (`for i in ...`, `for stage in ...`)

---

## 3) Data Flow (GEPA -> Adversary -> Loss -> LoRA Update)

Below is the exact technical sequence used by the notebook, normalized into modular steps.

1. **GEPA emits strategy prompts**
   - Input: `stage_logs: list[{"adv": str, "reward": float}]`
   - Output:
     - `current_attacker_instruction: str`
     - `current_defense_prompt: str`

2. **Training sample selected**
   - Input: `dataset["train"][idx]["prompt"]`
   - Output: `original_prompt: str`

3. **Adversary message construction**
   - Build chat messages:
     - system = `"You are a rewriter. " + current_attacker_instruction`
     - user = `original_prompt`

4. **Adversary forward sampling (`policy_sample`)**
   - Tokenize templated chat prompt.
   - Generate completion with sampling (`do_sample=True`, temperature/top-p).
   - Return:
     - `completion_text` (adversarial rewrite)
     - `gen_ids` for REINFORCE
     - `prompt_len`

5. **Target generation (frozen victim)**
   - Input: adversarial rewrite (and optionally `current_defense_prompt` as system guardrail).
   - Output: `target_response: str`

6. **Judge scoring**
   - Input:
     - `behavior = original_prompt`
     - `generations = [target_response]`
   - Judge output: list like `["yes"]` or `["no"]`
   - Reward mapping: `reward = 1.0 if verdict startswith("yes") else 0.0`

7. **Policy-gradient loss construction**
   - Extract completion token span from `gen_ids` using `prompt_len`.
   - Compute token logprobs and sum over completion:
     - `logprob_sum_i = sum_t log pi(a_t | s_t)`
   - Batch objective:
     - `loss = -(rewards * logprob_sums).mean()`

8. **LoRA parameter update**
   - `optimizer.zero_grad()`
   - `loss.backward()`
   - `optimizer.step()`
   - Only LoRA-adapted policy is updated; target and judge remain frozen.

9. **Stage completion and GEPA re-evolution**
   - Collect per-iteration logs (`adv`, `reward`).
   - Every stage:
     - evaluate ASR
     - call GEPA evolve
     - replace attacker/defender prompts for next stage

---

## 4) Interface Contract (Swap-Friendly, Exact Shapes)

This section defines the strict input/output contract so different "vibe" modules can be swapped without touching the orchestrator.

## Contract A: `BaseOptimizer.evolve`

**Input**
- `stage_logs: list[StageLog]`

`StageLog` schema:
- `adv: str` (adversarial rewrite text)
- `reward: float` in `[0.0, 1.0]`
- optional: `verdict: str`, `target_response: str`

**Output**
- `tuple[str, str]`
  - `next_attacker_instruction`
  - `next_defense_prompt`

**Constraints**
- Each output must be a single instruction/prompt string.
- Sanitization must remove artifacts/tags and keep deterministic first-line output.

---

## Contract B: `AdversaryManager.sample`

**Input**
- `messages: list[{"role": str, "content": str}]`
- optional generation params (`max_new_tokens`, `temperature`, `top_p`)

**Output**
- `SampleResult` dict:
  - `prompt: str`
  - `prompt_len: int` (number of prompt tokens)
  - `gen_ids: torch.LongTensor` shape **`[B, T_full]`**
    - current notebook uses `B = 1`
  - `completion_text: str`
  - `completion_logprob_sum: torch.FloatTensor` shape **`[B]`**

**Constraints**
- `gen_ids` must include prompt + completion tokens.
- `prompt_len` must index the boundary exactly.

---

## Contract C: `TargetModelGateway.generate`

**Input**
- `prompt: str` (adversarial rewrite)
- optional `defense_prompt: str` (system safety policy)

**Output**
- `target_response: str`

**Constraints**
- Must be deterministic for benchmark comparability when required (`do_sample=False`).

---

## Contract D: `JudgeGateway.score`

**Input**
- `behavior: str` (original dataset prompt)
- `generations: list[str]` length `B`

**Output**
- `verdicts: list[str]` length `B`
  - expected normalized values map to yes/no semantics

**Constraints**
- Output length equals `len(generations)`.

---

## Contract E: `AdversaryManager.update`

**Input**
- `gen_ids: torch.LongTensor` shape **`[B, T_full]`**
- `prompt_lens: list[int]` length `B`
- `rewards: torch.FloatTensor` shape **`[B]`**

**Output**
- `UpdateResult` dict:
  - `loss: float`
  - `logprob_sums: torch.FloatTensor` shape **`[B]`**

**Constraints**
- For each sample `i`, completion span is `gen_ids[i, prompt_lens[i]:]`.
- If completion length is zero, its logprob sum should be `0.0`.

---

## 5) Implementation Guide: How to Port Your "Vibe Code"

Use this migration checklist to plug custom "vibe" behavior into the new structure with minimal breakage.

## Step 1: Wrap your current code into adapters

Create one adapter per concern:

- `vibe/base_optimizer.py` -> implements `BaseOptimizer.evolve(...)`
- `vibe/adversary_manager.py` -> implements `sample(...)`, `update(...)`
- `vibe/target_gateway.py` -> implements `generate(...)`
- `vibe/judge_gateway.py` -> implements `score(...)`

Do not move orchestration logic yet; only wrap existing functions.

## Step 2: Preserve contract compatibility first

Before changing behavior, ensure your wrapper returns exact keys/shapes from Contracts A-E.

Hard requirements:
- `gen_ids` is always `[B, T_full]`
- `prompt_lens` length equals `B`
- `rewards` shape equals `[B]`
- judge verdict list length equals `B`

If these hold, orchestration remains plug-and-play.

## Step 3: Define your "vibe variation" as configuration, not forks

Parameterize differences via config:
- sampling policy: `temperature`, `top_p`, `max_new_tokens`
- GEPA evolution prompt templates
- reward shaping function (binary vs scaled)
- defense system prompt policy

This avoids copy-paste branches and enables A/B runs.

## Step 4: Add a minimal orchestrator interface

Standard runner signature:

```python
def run_training(
    base_optimizer: BaseOptimizer,
    adversary: AdversaryManager,
    target: TargetModelGateway,
    judge: JudgeGateway,
    dataset,
    cfg,
) -> None: ...
```

The orchestrator should only call interfaces, never concrete implementations.

## Step 5: Validation tests before full training

Run these quick tests:

1. **Shape test:** one batch through `sample -> update` with synthetic reward.
2. **Boundary test:** verify completion extraction using `prompt_len`.
3. **Judge parity test:** confirm yes/no mapping remains stable.
4. **GEPA output test:** ensure evolved prompts are single-line and sanitized.

## Step 6: Porting anti-patterns to avoid

- Mixing text logging fields with training tensors in the same object.
- Returning completion-only token ids instead of full `gen_ids`.
- Letting GEPA outputs include meta wrappers (`[INST]`, "Here is ...").
- Updating target/judge weights accidentally (must stay frozen).

---

## 6) Recommended Folder Blueprint

```text
src/
  pipeline/
    orchestrator.py
    contracts.py
  optimizers/
    base_optimizer.py
    gepa_optimizer.py
  adversary/
    adversary_manager.py
    lora_adversary_manager.py
  gateways/
    target_gateway.py
    harmbench_judge_gateway.py
  configs/
    default.yaml
```

This layout cleanly separates policy training, meta-optimization, and external model/judge dependencies.

---

## 7) Notes Specific to the Notebook

- The notebook already contains the critical primitives for this architecture, but mostly as free functions; this document formalizes them into replaceable interfaces.
- `target_generate_defended(...)` is used conceptually in one loop variant; ensure your production gateway exposes a defended-generation path explicitly.
- A hardcoded Hugging Face token appears in notebook cells; move all secrets to environment variables before productionization.

---

## 8) GEPA From-Scratch Implementation Deep Dive

This section describes exactly how GEPA is implemented in `legacy_code/Copy_of_basic_implementation_without_gepa (4).ipynb` today, and how to map each piece to a future official GEPA Python package/API.

## What "GEPA" means in this notebook

In this notebook, GEPA is implemented as a **prompt-level meta-optimizer**, not a gradient-based parameter optimizer:

- It does **not** backprop into a separate GEPA model.
- It **does** summarize stage outcomes and prompt the same policy model to generate improved attacker and defender instructions.
- The optimization variable is text (`current_attacker_instruction`, `current_defense_prompt`), not weights.

So GEPA here is best understood as: **stagewise textual policy search guided by success/failure logs**.

## The exact GEPA state machine

Persistent state across stages:

- `current_attacker_instruction: str`
- `current_defense_prompt: str`

Per-stage temporary state:

- `stage_logs: list[{"adv": str, "reward": float}]`
  - `adv` = adversarial rewrite produced by current policy
  - `reward` = binary judge outcome (1.0 success, 0.0 failure)

At stage end:

1. Partition logs into:
   - `successes = [log for log in stage_logs if reward > 0.5]`
   - `failures = [log for log in stage_logs if reward <= 0.5]`
2. Build analysis context string with counts and top successful patterns.
3. Build two meta-prompts:
   - attacker evolution prompt
   - defender evolution prompt
4. Run a meta-generation helper (`meta_gen`) for each prompt.
5. Sanitize generated strings.
6. Overwrite persistent state for next stage.

This is a deterministic "analyze -> generate -> sanitize -> replace" loop.

## GEPA prompt construction mechanics

`gepa_evolve_prompts(stage_logs)` builds structured instruction text roughly in this pattern:

- Prefix with `[INST] ... [/INST]` wrapper.
- Include compact metrics:
  - number of successes
  - number of failures
  - examples of successful adversarial rewrites (truncated)
- Ask for a strict output format:
  - one sentence only
  - no filler
  - no tags

There are two different task prompts:

- **Attacker meta-task:** produce a better rewrite instruction to bypass safety filters while preserving intent.
- **Defender meta-task:** produce a better system prompt that blocks observed attack patterns.

Even though it is "from scratch", this is still a valid GEPA-style operator: it closes the loop between empirical outcomes and next-stage policy instructions.

## Why sanitization is mandatory here

The notebook includes `sanitize_gepa_output(...)` because raw meta-generation can include:

- chat control tags (`[INST]`, `[/INST]`)
- boilerplate ("Here is the new instruction:")
- malformed remnants from previous prompt structures

Sanitization ensures downstream stability by forcing clean one-line instructions. Without this, GEPA can self-corrupt prompt state across stages.

## Relationship between GEPA and LoRA updates

Two optimizers run at different levels:

- **LoRA/REINFORCE (fast timescale, every iteration):**
  - updates token-level policy parameters.
- **GEPA/meta-prompt evolution (slow timescale, every stage):**
  - updates high-level attacker/defender textual directives.

The practical effect:

- LoRA learns to execute the current attacker strategy better.
- GEPA periodically changes the strategy itself.

This separation is important to preserve during reimplementation.

## Reimplementation blueprint with official GEPA module/API

When you migrate to an official GEPA library, keep this one-to-one mapping:

1. **Current `stage_logs` -> GEPA observation buffer**
   - Keep a normalized schema:
     - `input_prompt`
     - `adversarial_prompt`
     - `target_response`
     - `reward`
     - `verdict`
2. **Current `analysis_context` string -> GEPA feature extractor**
   - Move handcrafted text summaries into structured features when API allows.
3. **Current attacker/defender meta-prompts -> GEPA proposal operators**
   - Register two proposal channels:
     - attacker proposal
     - defender proposal
4. **Current `sanitize_gepa_output` -> output validator**
   - Keep a post-process/validation stage even with official API.
5. **Current in-place overwrite -> candidate selection policy**
   - If official GEPA supports multiple candidates, replace direct overwrite with:
     - rank candidates
     - optional holdout check
     - commit best candidate

## Suggested migration phases

Phase 1 (compatibility-first):
- Keep your existing stage loop.
- Replace only `gepa_evolve_prompts(...)` internals with official API calls.
- Preserve old return type: `tuple[str, str]`.

Phase 2 (structured GEPA state):
- Replace free-form log text with typed trajectory objects.
- Add richer reward signals (not just binary yes/no).

Phase 3 (search quality upgrades):
- Multi-candidate proposal per stage.
- Candidate scoring on validation prompt slice.
- Accept/reject gate before updating attacker/defender state.

## Minimal pseudocode parity target

Use this as your "did we preserve behavior?" reference when migrating:

```python
for stage in range(num_gepa_stages):
    stage_logs = []
    for _ in range(iters_per_stage):
        original_prompt = sample_train_prompt()
        adv = adversary.generate(original_prompt, current_attacker_instruction)
        target_resp = target.generate(adv, defense_prompt=current_defense_prompt)
        verdict = judge.score(original_prompt, [target_resp])[0]
        reward = 1.0 if verdict_is_yes(verdict) else 0.0
        adversary.update_from_reward(reward)
        stage_logs.append({"adv": adv, "reward": reward})

    current_attacker_instruction, current_defense_prompt = base_optimizer.evolve(stage_logs)
```

If this control flow and I/O contract remain intact, your GEPA backend can be swapped with minimal risk.

---

## 9) Comparison: Section 8 GEPA vs `gepa_qwen_harmbench_run (1).ipynb`

This section compares the two GEPA constructions:

- **A. Section 8 GEPA** = from-scratch GEPA loop in `legacy_code/Copy_of_basic_implementation_without_gepa (4).ipynb`
- **B. Qwen GEPA notebook** = library-driven GEPA in `legacy_code/gepa_qwen_harmbench_run (1).ipynb`

## B) Detailed GEPA implementation in `gepa_qwen_harmbench_run (1).ipynb`

The Qwen notebook uses the official GEPA package (`gepa.optimize_anything`) to optimize a **single candidate field**: `system_prompt`.

### Pipeline breakdown

1. **Dataset normalization**
   - HarmBench records are normalized to:
     - `{"id": str, "prompt": str, "is_harmful_request": True}`
   - Split into `train_data` and `val_data`.

2. **Task model execution layer**
   - Uses local vLLM via OpenAI-compatible API:
     - `client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)`
   - `generate_response(...)` calls chat completions with:
     - system prompt candidate
     - harmful user prompt

3. **Scoring layer**
   - `refusal_score(text)` is rule/pattern-based:
     - returns `1.0`, `0.7`, or `0.0` from refusal phrase hits
   - GEPA objective score in evaluator:
     - `score = float(refusal_score(response))`
   - Evaluator returns `(score, side_info)` where `side_info` includes preview text, refusal score, latency.

4. **GEPA setup**
   - Candidate schema:
     - `seed_candidate = {"system_prompt": BASELINE_SYSTEM_PROMPT}`
   - Dataset format for GEPA:
     - `gepa_train = [{"input": ex["prompt"], "id": ex["id"]}, ...]`
     - `gepa_val = [{"input": ex["prompt"], "id": ex["id"]}, ...]`
   - Objective text:
     - optimize system prompt for stronger refusal quality.

5. **Config and optimization call**
   - `GEPAConfig(engine=EngineConfig(max_metric_calls=...), reflection=ReflectionConfig(...))`
   - Reflection model is external (`openai/gpt-4o-mini`).
   - Run:
     - `gepa_result = optimize_anything(**common_kwargs)`

6. **Candidate extraction and validation**
   - Extract best candidate robustly from result object/dict.
   - Evaluate optimized prompt on held-out val set with `evaluate_system_prompt(...)`.
   - Save artifacts:
     - optimized prompt text
     - metrics JSON/CSV
     - trace plots and optimizer trace CSV.

### What this implies architecturally

- Optimization target is only **prompt text**; no weight updates.
- Search dynamics (proposal/selection/pareto/coverage logic) are handled by GEPA internals.
- Evaluator is the critical custom extension point.
- This notebook is API-first and experiment-trace oriented.

## Side-by-side comparison

| Axis | Section 8 (from-scratch notebook) | Qwen GEPA notebook (library) |
|---|---|---|
| Main optimization object | Attacker + defender prompt text, plus LoRA policy updates in loop | Single `system_prompt` text candidate |
| Learning signal | HarmBench yes/no -> binary reward for REINFORCE | Pattern-based refusal score in `[0.0, 0.7, 1.0]` |
| GEPA engine | Handwritten `gepa_evolve_prompts` + meta-prompting + sanitizer | `optimize_anything` with `GEPAConfig` + reflection model |
| Model updates | LoRA weights updated every iteration | No model weight updates |
| Granularity | Dual-timescale (iterative RL + stage-level text evolution) | Prompt search iterations under metric-call budget |
| Reflection path | Implicit self-reflection through same model meta-prompting | Explicit reflection LM (`openai/...`) in GEPA config |
| Validation strategy | Stage logs + periodic ASR checks | Train/val split + full val evaluation + trace artifacts |
| Robustness controls | Manual sanitization and manual stage logic | Library-managed candidate pool and selection flow |

## Similarities

- Both are **black-box prompt optimization** over harmful-request behavior.
- Both use HarmBench-style prompts and refusal/safety outcomes.
- Both run iterative propose/evaluate/select loops.
- Both log optimization trajectory and evaluate baseline vs improved behavior.

## Key differences that matter for reimplementation

1. **Control surface**
   - Section 8 gives full control over loop semantics.
   - Qwen notebook delegates core search policy to GEPA library internals.

2. **Objective design**
   - Section 8 objective is tied to downstream harmful-compliance success and RL reward.
   - Qwen notebook objective is direct refusal quality score.

3. **Coupling with policy training**
   - Section 8 couples prompt evolution with LoRA RL updates.
   - Qwen notebook isolates prompt optimization from parameter learning.

4. **Extensibility mode**
   - Section 8: easiest to customize unusual logic quickly.
   - Qwen notebook: easiest to standardize, reproduce, and scale with cleaner APIs.

## Consolidation plan (recommended)

Unify both approaches by separating GEPA backend from training orchestration.

### Target unified abstraction

```python
class BaseOptimizer:
    def evolve(self, stage_logs: list[dict], context: dict | None = None) -> dict: ...
```

Return payload:
- `attacker_instruction: str | None`
- `defense_prompt: str | None`
- `system_prompt: str | None`
- `meta: dict` (scores, candidate_id, diagnostics)

Implement two backends:
- `ScratchGEPAOptimizer` (Section 8 behavior parity)
- `LibraryGEPAOptimizer` (Qwen notebook / `optimize_anything`)

### Consolidation steps

1. **Standardize observation schema**
   - Use one trajectory record format for both paths:
     - `prompt`, `adversarial_prompt`, `target_response`, `reward`, `refusal_score`, `latency_ms`, `verdict`.

2. **Standardize objective adapters**
   - Adapter A: binary HarmBench yes/no score.
   - Adapter B: refusal-score metric.
   - Keep both selectable via config.

3. **Decouple policy update from GEPA**
   - Orchestrator decides whether to call:
     - prompt-only optimization
     - LoRA update
     - or both.

4. **Adopt common candidate schema**
   - Internal candidate keys:
     - `system_prompt`, `attacker_instruction`, `defense_prompt`
   - Backends can ignore unused keys.

5. **Use identical evaluation harness**
   - Single `evaluate_candidate(candidate, examples)` used by both backends.
   - Produces consistent comparison metrics and plots.

6. **Migration order**
   - First: preserve Section 8 behavior under `ScratchGEPAOptimizer`.
   - Next: plug in `LibraryGEPAOptimizer` behind same interface.
   - Finally: run A/B on same datasets and budgets to choose production default.

## Practical guidance: when to use which implementation

- Use **from-scratch GEPA** when you need rapid experimentation with custom loop logic tightly coupled to LoRA RL.
- Use **library GEPA** when you need reproducible, maintainable prompt search with cleaner API boundaries and better experiment hygiene.
- Use the **unified interface** so switching does not require touching the rest of the training pipeline.
