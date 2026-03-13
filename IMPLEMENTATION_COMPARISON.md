# Implementation Comparison: GEPA and Safety Optimization Pipelines

This document compares three implementations in this repository ecosystem:

1. **Implementation A**: Section 8 in `PROJECT_ARCHITECTURE.md`  
   (from-scratch GEPA + LoRA/REINFORCE loop in `legacy_code/Copy_of_basic_implementation_without_gepa (4).ipynb`)
2. **Implementation B**: Section 9 in `PROJECT_ARCHITECTURE.md`  
   (library-driven GEPA in `legacy_code/gepa_qwen_harmbench_run (1).ipynb`)
3. **Implementation C**: current codebase (`src/` + `scripts/`)  
   (modular scripts with GEPA variants, baseline eval, steering, and separate RL training loop)

---

## 1) Executive Summary

- **A (Section 8)** is a tightly coupled dual-timescale loop: LoRA policy learns every iteration; GEPA updates attacker/defender prompts every stage.
- **B (Section 9)** is prompt-only optimization using official GEPA APIs; no parameter updates, strong reproducibility and artifact hygiene.
- **C (current codebase)** is a modular toolkit with multiple runnable pipelines:
  - DSPy GEPA (`scripts/gepa_run.py`)
  - `optimize_anything` GEPA (`scripts/gepa_judge_optimize.py`)
  - activation steering (`scripts/steer_results.py`)
  - separate LoRA REINFORCE training path (`run_experiment.py` + `src/training/loop.py`)

The core opportunity is to unify C into one orchestrator and treat A/B as backend strategies under shared contracts.

---

## 2) Implementation A: Section 8 (From-Scratch GEPA + LoRA)

## Core idea

Implementation A performs co-optimization of:

- **token-level adversary behavior** (LoRA policy update via REINFORCE)
- **prompt-level strategy** (GEPA-like attacker/defender evolution from stage logs)

It uses a handcrafted state machine:

- persistent state: `current_attacker_instruction`, `current_defense_prompt`
- per-step data: adversarial rewrite, target response, judge verdict, reward
- per-stage data: `stage_logs` summarized into successes/failures

## How it works technically

1. Sample behavior prompt from HarmBench.
2. Build adversary chat context using current attacker instruction.
3. Generate adversarial rewrite (`policy_sample` / `adversary_output`).
4. Query target model using rewrite (with/without defense prompt).
5. Judge target response; map to binary reward.
6. Compute REINFORCE loss over generated completion tokens and update LoRA weights.
7. After `iters_per_stage`, call GEPA evolution function to update attacker/defender prompt text.
8. Repeat across stages.

## Strengths

- High experimental flexibility.
- Direct coupling between policy gradients and strategy evolution.
- Easy to inject custom heuristics in evolution and sanitization.

## Limitations

- Manual search/evolution logic; lower reproducibility.
- Text-sanitization fragility can contaminate state.
- Harder to standardize and maintain as team code.

---

## 3) Implementation B: Section 9 (Library GEPA Notebook)

## Core idea

Implementation B uses official GEPA (`gepa.optimize_anything`) to optimize a single text artifact: `system_prompt`.

## How it works technically

1. Normalize HarmBench examples and create train/val split.
2. Define evaluator:
   - run task model with candidate `system_prompt`
   - compute refusal score from response
   - return `(score, side_info)`
3. Configure GEPA:
   - `EngineConfig(max_metric_calls=...)`
   - `ReflectionConfig(reflection_lm="openai/...")`
4. Run `optimize_anything(...)`.
5. Extract best candidate.
6. Evaluate baseline vs optimized prompt on val set.
7. Save metrics, traces, and optimized prompt artifact.

## Strengths

- Strong API structure and reproducibility.
- Built-in optimization machinery (candidate proposals, selection behavior, budget controls).
- Cleaner integration for productionized prompt optimization.

## Limitations

- Prompt-only optimization by default; no LoRA/policy update coupling.
- Search internals are abstracted (less low-level control).
- Quality depends heavily on evaluator and scoring design.

---

## 4) Implementation C: Current `src/` + `scripts/` Codebase

Implementation C is not a single loop; it is a set of interoperable pipelines.

## C1) GEPA (optimize_anything) pipeline

File: `scripts/gepa_judge_optimize.py`

- Uses:
  - `src/data/harmbench_loader.py`
  - `src/evaluator/judge_vLLM.py`
  - `src/evaluator/refusal_score_from_judge.py`
- Baseline eval + GEPA optimization + optimized eval.
- Candidate schema: `{"system_prompt": ...}`.
- Reflection model via env-backed OpenAI key.
- Produces structured artifacts (baseline/optimized CSVs, trace CSV, summary JSON, prompt txt).

This is the closest production script to Implementation B.

## C2) GEPA (DSPy) pipeline

File: `scripts/gepa_run.py`

- Uses `dspy.GEPA` with `dspy.Predict(SafetySystem)`.
- Metric uses vLLM judge and returns score/feedback.
- Runs baseline and optimized evaluation; writes detailed CSV outputs and saved optimized program JSON.

## C3) Activation steering pipeline

Files:
- `scripts/steer_results.py`
- `src/steering_methods/activation_add.py`
- `scripts/launch_steering.sh`

Flow:

1. Build CAA positive/negative pairs.
2. Extract layer steering vectors.
3. Apply `ActivationAddition` hook at selected layer.
4. Evaluate with HarmBench + judge ASR.
5. Save per-example results and summary.

This is an orthogonal optimization method (activation-space intervention rather than prompt search).

## C4) LoRA REINFORCE training pipeline

Files:
- `src/adversary/policy.py` (`RedTeamPolicy`)
- `src/training/loop.py` (`TrainingLoop`)
- `run_experiment.py`

Flow:

1. Adversary generates jailbreak rewrite + log-prob.
2. Target responds (currently mock target in `run_experiment.py`).
3. Judge scores Yes/No.
4. Reward maps to scalar.
5. REINFORCE update on adversary LoRA policy.

Important: this loop currently has a TODO for GEPA evolution integration.

## C strengths

- Modular components (loader, judge, scoring, steering, adversary, loops).
- Multiple experimentation pathways from one codebase.
- Better separation of concerns than notebook-only setups.

## C limitations

- No single unified orchestrator across GEPA + LoRA + steering.
- Metric variants (binary reward vs refusal score) are split across scripts.
- Some scripts are prototype/stale (`scripts/main.py`) and not aligned with current architecture.

---

## 5) Complete Comparison Across A, B, C

| Axis | A: Section 8 from-scratch | B: Section 9 library GEPA | C: Current codebase |
|---|---|---|---|
| Primary optimization target | attacker + defender prompts + LoRA policy | single `system_prompt` | multiple: prompt-only GEPA, DSPy GEPA, steering, separate LoRA loop |
| GEPA engine | handcrafted `gepa_evolve_prompts` | `optimize_anything` + `GEPAConfig` | both `optimize_anything` and `dspy.GEPA` |
| Weight updates | yes (LoRA per iteration) | no | yes in RL path; no in prompt-only GEPA paths |
| Search granularity | stagewise prompt evolution + iterative RL | evaluator-call budgeted candidate search | script-dependent |
| Reflection mechanism | implicit/meta-prompted self-reflection | explicit reflection LM (`openai/...`) | explicit for optimize_anything path; metric feedback for DSPy |
| Judge signal | binary yes/no reward | refusal score (0/0.7/1 pattern-based in notebook) | binary reward and richer refusal-score adapter from judge explanation |
| Data split discipline | stage-driven training logs; periodic eval | explicit train/val split | explicit in GEPA scripts; baseline/steering use holdout slices |
| Artifact/log maturity | moderate, mostly notebook logs | strong notebook artifact output | strong in `gepa_judge_optimize.py`; mixed elsewhere |
| Extensibility | high but manual | high via API contracts | high if consolidated, currently fragmented |
| Production readiness | lower | medium-high | medium (good primitives, needs orchestration unification) |

---

## 6) Similarities and Differences

## Similarities

- All three optimize safety behavior against harmful prompts.
- All depend on external judge scoring.
- All can be framed as black-box optimization over model behavior.
- All already support artifact generation for analysis.

## Major differences

- **Coupling level**:
  - A tightly couples GEPA and LoRA.
  - B is prompt-only search.
  - C contains both styles but in separate scripts.
- **Optimization substrate**:
  - A: text + policy params.
  - B: text only.
  - C: text (GEPA), activations (steering), params (LoRA) depending on script.
- **Standardization**:
  - A least standardized.
  - B most API-standardized.
  - C partially standardized but currently split by entrypoint.

---

## 7) Integration/Unification Recommendations

## 7.1 Unified architecture target

Create one orchestrator and treat each strategy as a pluggable backend.

Recommended interfaces:

```python
class CandidateManager:
    def propose(self, observations: list[dict], state: dict) -> dict: ...
    def select(self, candidates: list[dict], eval_results: list[dict]) -> dict: ...

class PolicyUpdater:
    def sample_attack(self, behavior: str, candidate: dict) -> dict: ...
    def update(self, rollout_batch: list[dict]) -> dict: ...

class Evaluator:
    def evaluate(self, candidate: dict, behavior: str) -> dict: ...
```

Candidate schema (single canonical payload):

- `system_prompt: str | None`
- `attacker_instruction: str | None`
- `defense_prompt: str | None`
- `steering: {"layer": int, "coefficient": float, "vector_path": str} | None`
- `meta: dict`

## 7.2 Backends to support

- `ScratchGEPABackend` (A parity)
- `LibraryGEPABackend` (B parity)
- `DSPyGEPABackend` (C2)
- `SteeringBackend` (C3)
- `LoRAPolicyBackend` (C4)

The orchestrator composes backends via config:

- prompt-only run
- LoRA-only run
- prompt + LoRA hybrid
- prompt + steering
- full tri-modal hybrid

## 7.3 Unified metric layer

Standardize on one scorer object with two modes:

- `binary_jailbreak_reward` (0/1)
- `graded_refusal_score` (0/0.7/1 or calibrated variant)

Store both in every record:

- `reward_binary`
- `refusal_score`
- `judge_label`
- `judge_explanation`

This removes metric drift across scripts.

## 7.4 Unified run schema and artifacts

Define a shared run directory format:

- `run_config.yaml`
- `observations.parquet` (or CSV)
- `candidates.jsonl`
- `metrics_summary.json`
- `trace_optimizer.csv`
- `final_candidate.json`

All pipelines should emit this structure, regardless of backend.

## 7.5 Migration plan

1. **Phase 1: Interface extraction**
   - Refactor existing scripts to call shared evaluator/data/judge modules only.
2. **Phase 2: Common orchestrator**
   - Add one runner that accepts backend type and mode.
3. **Phase 3: A/B parity tests**
   - Reproduce Section 8 and Section 9 results through the unified runner.
4. **Phase 4: Hybrid experiments**
   - Add LoRA+GEPA and GEPA+steering combined modes under one config surface.

---

## 8) Practical Next Steps for This Repository

1. Move GEPA-specific logic from `scripts/gepa_judge_optimize.py` into `src/optimizers/library_gepa.py`.
2. Implement `src/optimizers/scratch_gepa.py` to preserve Section 8 behavior.
3. Add `src/pipeline/orchestrator.py` with config-driven backend selection.
4. Wire `run_experiment.py` to optionally call GEPA backend each epoch/stage.
5. Keep `scripts/` as thin CLI entrypoints that call shared orchestrator APIs.
6. Mark `scripts/main.py` as deprecated or fix/remove it to avoid architecture confusion.

---

## 9) Bottom Line

- If your near-term goal is reproducible prompt optimization, use **Implementation B/C1** as the baseline.
- If your goal is research into co-adaptation (strategy + policy weights), preserve **Implementation A** semantics inside a backend.
- The best long-term path is **Implementation C unified**: one orchestration layer, multiple interchangeable optimization backends, one standardized evaluation contract.

