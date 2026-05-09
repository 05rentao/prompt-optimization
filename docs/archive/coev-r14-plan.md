# R14 — Full Co-Evolution with RLOO + Length Penalty — Change Plan

Source spec: `coev-experiments-final.md`

## R14 in one sentence

Run the **CoEV v2** co-evolutionary loop (staged adversary RL updates alternating with dual-role GEPA), but with the adversary updated via **RLOO** and its reward shaped by the **length penalty** (`w=0.2`, `min_tokens=50`) — matching the R11 adversary-only configuration — so we can compare co-evolutionary robustness against the adversary-only baseline.

## Target configuration (from the spec)

| Field | Value | Source line |
|---|---|---|
| Adversary | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | adversarial setting |
| Target | `meta-llama/Llama-3.1-8B-Instruct` | adversarial setting |
| Judge | `cais/HarmBench-Mistral-7b-val-cls` | adversarial setting |
| Policy | RLOO | R14 + adversarial setting |
| Batch (K rollouts per step) | 4 | adversarial setting |
| `length_penalty_weight` | 0.2 | adversarial setting |
| `length_penalty_min_tokens` | 50 | inherited from R11 (`configs/adversary_rloo_length_penalty.yaml`) |
| Seed | 42 | adversarial setting |
| Adversary iterations (total) | **60 or 120** — spec conflict (see Q1) | R14 says 60; adversarial setting says 120 |
| Defender | GEPA (dual-role, built-in) | R14 |
| Eval | judge-based, ASR + refusal + aggregate per stage boundary | R14 |

## Current codebase support

### What's already there (no change needed)

- **`runs/coev_v2_run.py`** — staged adversary RL + dual-role GEPA alternation.
- **RLOO update** — `--adversary-policy rloo` hits `rloo_update_batch_sgd` inside the stage loop (lines 889–897).
- **Stage-boundary evaluation** — `coev_v2_stage_metrics.csv` emits rows for `pre_evolution`, `attacker_gepa_best`, `defender_gepa_best`, and `gepa_seconds` per stage (lines 931–991).
- **Per-stage evolved prompts** — `optimizer_trace_attacker.csv` / `optimizer_trace_defender.csv` capture GEPA candidate trajectories; `optimized_attacker_instruction.txt` / `optimized_defense_prompt.txt` hold the final versions.
- **Run manifest** — `run_manifest.json` with `mode: "coev_v2_rloo"`, full config snapshot, endpoints, seed, dataset, models.
- **Length-penalty reference implementation** — already in `runs/adversary_run.py::shape_reward_with_length_penalty` (lines 193–212) and wired for R11 via `configs/adversary_rloo_length_penalty.yaml`.

### What's missing for an exact R14 run

1. **Length penalty is NOT wired into `runs/coev_v2_run.py`.** Searching for `length_penalty` in `runs/coev_v2_run.py` returns zero matches. The reward out of `multi_query_reward(...)` is passed straight to the RLOO update with no shaping (lines 822–897).
2. **`runs.coev_v2` YAML block in `configs/default.yaml` has no `length_penalty_*` keys**, so even adding CLI flags alone would not persist defaults for teammates.
3. **No CoEV YAML preset** mirroring R11's `configs/adversary_rloo_length_penalty.yaml` (identical models + new coev_v2 knobs).

## Proposed code changes

Minimal, localized, backward-compatible (default `length_penalty_weight=0.0` → no behavior change for other users).

### Change 1 — Promote `shape_reward_with_length_penalty` to `src/run_pipeline.py`

**Why:** both `runs/adversary_run.py` and `runs/coev_v2_run.py` need the identical reward-shaping function. Copy-pasting would drift; cross-`runs/` imports are an anti-pattern per `README.md`'s "plumbing vs experiment logic" section.

**Edit `src/run_pipeline.py`:** add

```python
import torch

def shape_reward_with_length_penalty(
    base_reward: float,
    gen_ids: torch.Tensor,
    prompt_len: int,
    length_penalty_weight: float,
    length_penalty_min_tokens: int,
) -> float:
    """shaped = base_reward * (1 - w) + length_ratio * w, short-circuits when w <= 0."""
    if length_penalty_weight <= 0.0:
        return base_reward
    completion_tokens = gen_ids.shape[-1] - prompt_len
    length_ratio = min(completion_tokens / max(length_penalty_min_tokens, 1), 1.0)
    return base_reward * (1.0 - length_penalty_weight) + length_ratio * length_penalty_weight
```

**Edit `runs/adversary_run.py`:**
- Remove the local definition (lines 193–212).
- Import from `src.run_pipeline` (alongside existing `build_prompt_pool`, `compute_reward_and_verdict`, etc.).
- No behavioral change.

### Change 2 — Wire length penalty into `runs/coev_v2_run.py`

**Add CLI flags** in `parse_args(...)` (after `--adversary-reinforce-batch-size`):

```python
parser.add_argument(
    "--length-penalty-weight",
    type=float,
    default=run_defaults.get("length_penalty_weight", 0.0),
    help="Weight for length-based reward shaping (0.0 disables).",
)
parser.add_argument(
    "--length-penalty-min-tokens",
    type=int,
    default=run_defaults.get("length_penalty_min_tokens", 50),
    help="Minimum completion tokens for full length bonus.",
)
```

**Import** `shape_reward_with_length_penalty` from `src.run_pipeline`.

**Apply shaping in the rollout loop** (`_run_training_and_finalize`). Both the `use_rs` branch (lines 810–840) and the non-RS branch (lines 841–866) call `multi_query_reward(...)`. After each call, replace

```python
batch_rewards.append(reward)
```

with

```python
shaped = shape_reward_with_length_penalty(
    reward, sample["gen_ids"], sample["prompt_len"],
    args.length_penalty_weight, args.length_penalty_min_tokens,
)
batch_rewards.append(shaped)
```

**Log it in the training row** (lines 910–929): add `"length_penalty_weight": args.length_penalty_weight` and `"mean_completion_tokens": sum(g.shape[-1] - p for g, p in zip(batch_gen_ids, batch_prompt_lens)) / max(len(batch_gen_ids), 1)` — same keys as `adversary_run.py` so R14's training CSV is column-compatible with R11's for downstream diffing.

**Manifest / metrics JSON:** add `length_penalty_weight` and `length_penalty_min_tokens` to the `config` block (line 422) and to `manifest.extra` (line 525) for auditability.

### Change 3 — Add defaults to `configs/default.yaml`

Under `runs.coev_v2` (around lines 115–139), add:

```yaml
    length_penalty_weight: 0.0        # 0 = off (backward compatible with existing coev_v2 runs)
    length_penalty_min_tokens: 50
```

### Change 4 — New R14 config preset: `configs/coev_r14_rloo_length_penalty.yaml`

Mirrors `configs/adversary_rloo_length_penalty.yaml`'s `global` / `runtime` / `shared_generation` blocks exactly (so models, judge, seed, attacker instruction, and target system prompt are identical to R11), then overrides `runs.coev_v2` with:

```yaml
runs:
  coev_v2:
    train_size: 100
    val_size: 20
    results_dir: "results/coev_r14_rloo_length_penalty"
    training_csv_name: "coev_v2_training_log.csv"
    adversary_policy: rloo
    adversary_prompt: default  # ignored when initial_attacker_instruction is set in shared_generation
    target_queries: 1
    rs_budget: 5
    rs_min_successes: 0
    adversary_reinforce_batch_size: 4
    target_max_workers: 16

    # --- total adversary iterations = stages * iters_per_stage ---
    # See Q1 / Q2 — needs answer before finalising these two numbers.
    stages: ???
    iters_per_stage: ???
    eval_every_stages: 1

    train_slice_end: 100
    eval_slice_start: 100
    eval_slice_end: 120

    lr: 0.00001
    weight_decay: 0.01
    max_metric_calls: 50
    eval_method: "judge"

    length_penalty_weight: 0.2
    length_penalty_min_tokens: 50
```

## Launch command (once Q1–Q3 are answered)

```bash
# Ensure vLLM is serving Llama-3.1-8B-Instruct at REFLECTION_VLLM_BASE_URL
export HF_TOKEN=hf_...

PROMPT_OPT_CONFIG_PATH=configs/coev_r14_rloo_length_penalty.yaml \
  uv run python runs/coev_v2_run.py \
    --mode coev \
    --adversary-policy rloo \
    --save-dir adapters/coev_r14_rloo_length_penalty
```

Or via the unified runner (same effect, needs `scripts.unified_runner.coev_v2_rloo_results_dir` aligned):

```bash
PROMPT_OPT_CONFIG_PATH=configs/coev_r14_rloo_length_penalty.yaml \
  uv run python scripts/run_unified_experiment.py --mode coev_v2_rloo
```

## Artifacts produced (under `results/coev_r14_rloo_length_penalty/`)

- `coev_v2_run_metrics.json` — baseline vs optimized aggregate metrics + config snapshot.
- `baseline_vs_optimized_metrics.csv`, `baseline_eval_outputs.csv`, `optimized_eval_outputs.csv`.
- `coev_v2_training_log.csv` — per-adversary-iteration reward/loss/verdict/batch/`length_penalty_weight`/`mean_completion_tokens`.
- **`coev_v2_stage_metrics.csv`** — stage-boundary ASR / refusal / aggregate (`phase` ∈ {`pre_evolution`, `attacker_gepa_best`, `defender_gepa_best`, `gepa_seconds`}). This is the oscillation / arms-race trace R14 asks for.
- `optimizer_trace_attacker.csv` / `optimizer_trace_defender.csv` — per-stage GEPA candidate trajectories.
- `optimized_attacker_instruction.txt`, `optimized_defense_prompt.txt`, `coev_v2_optimized_prompts.json` — final evolved prompts.
- `plot_baseline_vs_optimized.png`, `plot_optimization_trajectory.png`, `plot_asr_vs_iterations.png`, `plot_refusal_vs_iterations.png`.
- `run_manifest.json` (`mode: "coev_v2_rloo"`).
- Optional LoRA adapter under `--save-dir`.

## Decisions taken (from the AskQuestion round)

| # | Question | Answer |
|---|---|---|
| Q1 | Iteration total | **60 total adversary updates** |
| Q2 | Stage split | **6 stages × 10 iters/stage** (6 GEPA alternation boundaries) |
| Q3 | Per-iteration prompt saving | **Per-stage GEPA + attacker/defense prompt columns in every inner training row** (already emitted at `training_rows` so R14 gets free per-inner-iteration visibility into which prompt the rollout used) |
| Q4 | Length-penalty helper placement | **Duplicate in `coev_v2_run.py`** (no `src/` refactor; smaller blast radius) |
| Q5 | GEPA budget | Keep **defaults** (`max_metric_calls=50`, `gepa_max_tokens=256`, `gepa_temperature=0.7`) |
| Q6 | vLLM readiness | vLLM is already serving Llama-3.1-8B at `127.0.0.1:8001` |
| Q7 | Baseline eval | **Add `--skip-baseline-eval` flag to coev_v2** so R14 can reuse R11 baseline numbers |

## Changes actually made

### `runs/coev_v2_run.py`
1. Added local `shape_reward_with_length_penalty(...)` helper (≈ line 173), duplicated from `runs/adversary_run.py` so the rollout loop can shape rewards without an `src/` refactor.
2. Applied the shaping to every `reward` returned by `multi_query_reward(...)` in both the `use_rs` and non-RS rollout branches. The rejection-sampling success threshold also uses the shaped reward (consistent with how `adversary_run.py` treats it).
3. Added three CLI flags: `--length-penalty-weight`, `--length-penalty-min-tokens`, `--skip-baseline-eval`. Defaults come from `runs.coev_v2` YAML keys and stay backward compatible (`0.0`).
4. Extended the per-iteration training row with `length_penalty_weight`, `length_penalty_min_tokens`, and `mean_completion_tokens` columns. `attacker_instruction` and `defense_prompt` were already logged per inner iteration, which satisfies Q3.
5. Propagated `length_penalty_weight`, `length_penalty_min_tokens`, `skip_baseline_eval` into the metrics JSON `config` block and the `run_manifest.json` `extra` block for auditability.
6. `main()` now short-circuits the baseline `_eval_suite` call when `--skip-baseline-eval` is set, mirroring `runs/adversary_run.py`.

### `configs/default.yaml`
Added backward-compatible defaults under `runs.coev_v2`:
```yaml
    length_penalty_weight: 0.0
    length_penalty_min_tokens: 50
```

### `configs/coev_r14_rloo_length_penalty.yaml` (new)
Mirrors R11's `configs/adversary_rloo_length_penalty.yaml` (same models, judge, attacker instruction, target system prompt, seed) but populates `runs.coev_v2` with the R14 schedule (6×10, RLOO, length_penalty=0.2, min_tokens=50, lr=1e-5, GEPA default budget).

### Files deliberately NOT changed

- `runs/adversary_run.py` — per Q4, the existing local copy of `shape_reward_with_length_penalty` stays; no cross-imports. A future cleanup pass could promote both copies to `src/run_pipeline.py`.
- `src/` — no runtime plumbing touched; all edits sit in `runs/` + `configs/` per the "optional logic vs plumbing" rule in `README.md`.

## Launch command

```bash
# 1) Verify vLLM is serving Llama-3.1-8B-Instruct at 127.0.0.1:8001 (Q6 confirmed ready).
#    If it is on a different URL, prepend:  REFLECTION_VLLM_BASE_URL=http://host:port/v1

export HF_TOKEN=hf_...

PROMPT_OPT_CONFIG_PATH=configs/coev_r14_rloo_length_penalty.yaml \
  uv run python runs/coev_v2_run.py \
    --mode coev \
    --adversary-policy rloo \
    --skip-baseline-eval \
    --save-dir adapters/coev_r14_rloo_length_penalty
```

If you prefer to run the full baseline instead of reusing R11's, drop `--skip-baseline-eval` (~40 min added).

## Backward-compatibility check

- `length_penalty_weight` and `length_penalty_min_tokens` default to `0.0` / `50`. When `0.0` the shaping helper short-circuits, so pre-existing `coev_v2` and `coev_v2_rloo` runs produce identical outputs (same loss, same reward).
- `--skip-baseline-eval` defaults to `False`, so default behavior is unchanged.
- Three new keys in the training CSV and two new keys in `run_manifest.json.extra` — additive only, no column renames or removals.

## Follow-ups (not done, flagged for later)

- Unit test around `shape_reward_with_length_penalty` (currently uncovered by `tests/test_policy_gradient.py`).
- If length-penalty adoption grows, promote the helper into `src/run_pipeline.py` to remove the duplicate across `adversary_run.py` and `coev_v2_run.py`.
- Consider adding `coev_v2_rloo_length_penalty` as a first-class mode in `scripts/run_unified_experiment.py` so it does not need `PROMPT_OPT_CONFIG_PATH`.
