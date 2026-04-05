# Theme 2: Adversary setup and experiments

Working note for HarmBench-style adversary sweeps: single-shot rewrites (`adversary_run`), decomposition attacks (`adversary_v2`), and optional vanilla baselines (`coev_v2_run`). This project uses **`uv`**; run entrypoints from the **repo root** with `uv run python …`.

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| **`HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`** | Hugging Face access for HarmBench data and judge/adversary weights. |
| **Target runtime** | Usually OpenAI-compatible HTTP (e.g. vLLM) for the victim; see `runtime.reflection` / `REFLECTION_VLLM_*` in config and [`src/runtime/README.md`](src/runtime/README.md). |
| **Unique `--results-dir`** | One directory per experiment so metrics and `adapters/` never overwrite each other. |

Default behaviors CSV: `data/harmbench_behaviors.csv` via `runs.adversary.harmbench_csv_path` in [`configs/default.yaml`](configs/default.yaml) (override with `--harmbench-csv-path` where supported).

---

## 2. Which script to use

| Script | Role |
| --- | --- |
| [`runs/adversary_run.py`](runs/adversary_run.py) | **Single-shot:** rewriter instruction → one adversary completion per behavior → target → judge/heuristic. Main surface for **prompt presets** and policy sweeps. |
| [`runs/adversary_v2.py`](runs/adversary_v2.py) | **Decomposition:** one adversary completion that lists sub-questions → **one target completion per sub-question** → concatenate → judge/heuristic. Same **policy / RS / LoRA** options as v1 when `--mode train`. |
| [`runs/coev_v2_run.py`](runs/coev_v2_run.py) | **Optional control:** fixed attacker + defense prompts, staged training, no Theme 2 rewriter presets. |

**Config:** `shared_generation` → `runs.<name>` in `configs/default.yaml`. **`adversary_v2`** merges **`runs.adversary`** + **`runs.adversary_v2`** (v2 wins on conflicts); `harmbench_csv_path` often comes from `runs.adversary` if omitted under v2.

**Quick comparison**

| | `adversary_run` | `adversary_v2` |
| --- | --- | --- |
| **Default `--mode`** | `train` | `eval` |
| **Rewriter** | Named presets + optional `--attacker-instruction` | Built-in Lead Safety Architect template with `{{behavior}}` (or `--adversary-system-prompt-file`; legacy files omit the placeholder) |
| **Target calls per behavior** | 1 | 1 + (number of parsed sub-questions) |
| **v1-only flags** | `--adversary-prompt` | — |
| **v2-only flags** | — | `--max-subquestions`, `--target-sub-workers`, `--adversary-system-prompt-file`, `--adversary-temperature`, `--adversary-top-p` |
| **Shared training flags** | (via `runs.adversary` / merged defaults) | policy, RS, `iterations`, `lr`, `--no-finetune`, HF + CSV sizes |

---

## 3. Data splits (shared)

- **HF train pool:** `hf_train_size` rows (shuffle `seed`) from the Hugging Face HarmBench split — used when **`--mode train`** and you fine-tune.
- **CSV val / test:** `load_harmbench_csv_val_test_splits`: **`csv_val_size`** (periodic val ASR while training), **`csv_test_size`** (baseline + final test), **`csv_seed`**.

Val/test prompts **always** come from the CSV (no HF-only eval slice). For comparable rows, keep **`csv_seed`**, **`csv_test_size`**, and **`harmbench_csv_path`** fixed.

---

## 4. Recommended workflow

1. Freeze **`csv_seed`** / **`csv_test_size`** for reporting.
2. **Phase A** — `adversary_run`, `--no-finetune`, prompt presets.
3. **Phase B** — Fine-tune winners; REINFORCE / RLOO / rejection sampling.
4. **Phase C** — Optional hypers (`iterations`, `lr`, RS, batch size).
5. **Phase D** — `adversary_v2` on the **same** CSV slice as A/B.
6. **Phase E** — Optional `coev_v2_run` or eval-only.

Inventory tables: [§7](#7-experiment-inventory). **Deliverable:** one sheet with repro + ASR + **Winners**.

---

## 5. Streamlined entrypoints (same outline)

The two subsections below use the **same headings** where it applies: pipeline → modes & data → CLI → policy → **script-specific knobs** → **command switches** → throughput → artifacts pointer.

### 5.1 `adversary_run.py` (single-shot)

**Pipeline (per behavior)**  
HF/CSV behavior text → adversary **rewrite** (`adversary_rewrite_sample` with rewriter instruction) → **one** target completion → reward from judge or heuristic → (if training) policy-gradient step on LoRA.

**Modes & data**

| `--mode` | Training data | CSV usage |
| --- | --- | --- |
| `train` | HF pool `hf_train_size` | Val: periodic ASR; test: baseline + final |
| `eval` | *(loads data; no optimizer steps)* | Test slice for eval only |

**CLI switches (common)** — `uv run python runs/adversary_run.py --help`

| Flag | Purpose |
| --- | --- |
| `--results-dir` | Output directory (use a **unique** path per run; default from `runs.adversary` / `runs.adversary_v2` YAML if omitted). |
| `--mode` | `train` \| `eval`. |
| `--no-finetune` | `train` only: skip LoRA (prompt screen at init weights). |
| `--hf-train-size`, `--csv-val-size`, `--csv-test-size`, `--csv-seed`, `--harmbench-csv-path` | Dataset sizing and CSV shuffle. |
| `--eval-method` | `judge` \| `heuristic`. |
| `--adversary-policy`, `--adversary-reinforce-batch-size`, `--rs-min-successes`, `--rs-budget` | Policy / batch / rejection sampling. |
| `--target-max-workers` | Parallel **target** HTTP calls across **behaviors** (pipelining). |
| `--attacker-instruction`, `--target-system-prompt`, `--device` | Overrides. |

**Training hyperparameters** (`iterations`, `lr`, `eval_every`, `weight_decay`) are read from **`configs/default.yaml`** under `runs.adversary` — this script does **not** define `--iterations` / `--lr` / `--eval-every` on the CLI.

**v1-only:** `--adversary-prompt` (`default` … `fictional`).

**Policy (pick one run)** — REINFORCE (optional `--adversary-reinforce-batch-size K`), RLOO (`--adversary-policy rloo`, not with RS), rejection sampling (`--rs-min-successes` > 0, reinforce only).

**Commands — switching modes**

```bash
# Eval only (no training steps)
uv run python runs/adversary_run.py --mode eval --results-dir results/week-11/adv_eval

# Train, prompt-only (no LoRA)
uv run python runs/adversary_run.py --mode train --no-finetune \
  --adversary-prompt persona --results-dir results/week-11/adv_prompt_only

# Train + fine-tune (defaults from YAML)
uv run python runs/adversary_run.py --mode train --results-dir results/week-11/adv_ft

# Policy switches (examples)
uv run python runs/adversary_run.py --mode train --adversary-policy rloo --results-dir results/week-11/adv_rloo
uv run python runs/adversary_run.py --mode train --adversary-policy reinforce \
  --rs-min-successes 1 --rs-budget 5 --results-dir results/week-11/adv_rs

# Preset sweep (--no-finetune)
for PROMPT in default persona decompose academic fictional; do
  uv run python runs/adversary_run.py --mode train --no-finetune \
    --adversary-prompt "$PROMPT" --results-dir "results/week-11/adv_matrix_${PROMPT}"
done
```

**Throughput:** `--target-max-workers` affects how many target requests overlap **across examples**; raising it past GPU saturation can hurt latency (see [§8](#8-throughput-and-target-concurrency)).

**Artifacts (summary):** metrics JSON, before/after eval CSVs, training log + plots when fine-tuning, `adapters/` when LoRA runs. **Full list:** [Appendix A](#appendix-a-adversary_runpy).

---

### 5.2 `adversary_v2.py` (decomposition)

**Pipeline (per behavior)**  
Behavior → adversary emits **numbered sub-questions** (`START_DECOMPOSITION` … `END_DECOMPOSITION`) → parse / cap (`--max-subquestions`) → **one target completion per sub-question** (ordered; parallelized with `--target-sub-workers` when the HTTP target supports it) → concatenate → `evaluate_outputs` on **original behavior** vs concatenated text (judge uses a decomposition-specific template).

**Modes & data**

| `--mode` | Training data | CSV usage |
| --- | --- | --- |
| `eval` (default) | — | Test slice only (`val_size=0`, `test_size=csv_test_size`, `csv_seed`) |
| `train` | HF `hf_train_size` | Val + test from CSV (same split logic as v1) |

**CLI switches (common)** — `uv run python runs/adversary_v2.py --help`

| Flag | Purpose |
| --- | --- |
| `--results-dir` | Output directory (use a **unique** path per run; default from `runs.adversary` / `runs.adversary_v2` YAML if omitted). |
| `--mode` | `train` \| `eval` (**default `eval`**). |
| `--no-finetune` | `train` only: skip LoRA. |
| `--hf-train-size`, `--csv-val-size`, `--csv-test-size`, `--csv-seed`, `--harmbench-csv-path` | Same idea as v1 (`harmbench` path required; often from merged YAML). |
| `--eval-method` | `judge` \| `heuristic`. |
| `--adversary-policy`, `--adversary-reinforce-batch-size`, `--rs-min-successes`, `--rs-budget` | Same policy family as v1 when training. |
| `--max-new-tokens`, `--target-max-new-tokens` | Adversary decomposition budget; per-sub target budget. |

**v2-only**

| Flag | Purpose |
| --- | --- |
| `--max-subquestions` | Cap parsed sub-questions (default often 12). |
| `--target-sub-workers` | Parallelism for **sub-question** target calls. |
| `--adversary-system-prompt-file` | Replace built-in template; use `{{behavior}}` in the file to inject the HLD per line (else behavior stays in the user turn). |
| `--adversary-temperature`, `--adversary-top-p` | Sampling for decomposition text (target stays temperature 0 in rollout). |

**Training hyperparameters** (`iterations`, `lr`, `eval_every`, `weight_decay`) come from merged YAML under **`runs.adversary_v2`** (and inherited `runs.adversary` defaults) — no `--iterations` / `--lr` CLI flags here either.

**Policy:** Same REINFORCE / RLOO / rejection sampling rules as v1 when `--mode train` and `finetune` is on.

**Commands — switching modes**

```bash
# Eval only (default): outputs + metrics JSON
uv run python runs/adversary_v2.py --results-dir results/week-11/v2_eval

# Train + fine-tune
uv run python runs/adversary_v2.py --mode train --results-dir results/week-11/v2_ft

# Train, prompt-only / no LoRA
uv run python runs/adversary_v2.py --mode train --no-finetune --results-dir results/week-11/v2_prompt_only

# Policy switches (same pattern as v1)
uv run python runs/adversary_v2.py --mode train --adversary-policy rloo --results-dir results/week-11/v2_rloo
uv run python runs/adversary_v2.py --mode train --adversary-policy reinforce \
  --rs-min-successes 1 --rs-budget 5 --results-dir results/week-11/v2_rs

# Eval-method + decomposition knobs (examples)
uv run python runs/adversary_v2.py --eval-method heuristic --results-dir results/week-11/v2_heuristic
uv run python runs/adversary_v2.py --max-subquestions 8 --target-sub-workers 4 --results-dir results/week-11/v2_ablate
```

**Throughput:** `--max-subquestions` sets **load per behavior**; `--target-sub-workers` sets overlap among sub-question calls. See [§8](#8-throughput-and-target-concurrency).

**Artifacts (summary):** `eval` → `adversary_v2_outputs.csv` + `adversary_v2_metrics.json`. `train` → full suite including `adversary_v2_run_metrics.json`, training log, plots, `adapters/` when fine-tuning. **Full list:** [Appendix B](#appendix-b-adversary_v2py).

---

## 6. `coev_v2_run.py` (optional)

Staged REINFORCE/RLOO (and optional RS) with **fixed** YAML/CLI attacker + defense strings — no `--adversary-prompt`. Use for a stable non–Theme-2 baseline. Details: [Appendix C](#appendix-c-coev_v2_runpy).

---

## 7. Experiment inventory

**Legend:** Status → `done` / `skip` / `failed`. ASR: judge unless noted `heuristic`.

### Phase A — Prompt presets (`adversary_run`, no weight updates)

| ID | Status | Experiment | Command (minimal) | `--results-dir` (example) | `csv_seed` | `csv_test_size` | refusal_rate | asr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | complete | `default` | `uv run python runs/adversary_run.py --mode train --no-finetune --adversary-prompt default` | `results/adversary_default` |  |  | 0.76 | 0.24
| A2 | complete | `persona` | `--adversary-prompt persona` | `results/adversary_persona` |  |  | 0.76 | 0.24
| A3 | complete | `decompose` | `--adversary-prompt decompose` | `results/adversary_decompose` |  |  | 0.92 | 0.08
| A4 | complete | `academic` | `--adversary-prompt academic` | `results/adversary_academic` |  |  | 0.76 | 0.24
| A5 | complete | `fictional` | `--adversary-prompt fictional` | `results/adversary_fictional` |  |  | 0.7 | 0.3

### Phase B — Policy comparison (fine-tune)

| ID | Status | Policy | `adversary_prompt` | Command hints | `--results-dir` (example) | `csv_seed` | Test ASR (baseline) | Test ASR (final) | Best val ASR | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | | REINFORCE (default batch) | WINNER1 | `--adversary-policy reinforce` | `results/week-11/B1_reinforce_w1` | | | | | |
| B2 | | REINFORCE | WINNER2 | | `results/week-11/B2_reinforce_w2` | | | | | |
| B3 | | REINFORCE, batch 4 | WINNER1 | `--adversary-reinforce-batch-size 4` | `results/week-11/B3_reinforce_b4_w1` | | | | | |
| B4 | | RLOO | WINNER1 | `--adversary-policy rloo` | `results/week-11/B4_rloo_w1` | | | | | |
| B5 | | RLOO | WINNER2 | | `results/week-11/B5_rloo_w2` | | | | | |
| B6 | | Rejection sampling | WINNER1 | `--rs-min-successes 1 --rs-budget 5` | `results/week-11/B6_rs_w1` | | | | | |

### Phase C — Hyper sweeps (optional)

| ID | Status | What varies | `--results-dir` (example) | Test ASR (final) | Notes |
| --- | --- | --- | --- | --- | --- |
| C1 | | `iterations` (YAML only) | `results/week-11/C1_iters_*` | | |
| C2 | | `lr` / `weight_decay` | `results/week-11/C2_lr_*` | | |
| C3 | | `rs_budget` / `rs_min_successes` | `results/week-11/C3_rs_*` | | |
| C4 | | `adversary_reinforce_batch_size` | `results/week-11/C4_batch_*` | | |
| C5 | | Custom `--attacker-instruction` | `results/week-11/C5_custom_instr` | | |

### Phase D — Decomposition (`adversary_v2`)

| ID | Status | Experiment | Command hints | `--results-dir` (example) | `csv_seed` | `csv_test_size` | Test ASR | Judge vs heuristic | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | | Default + judge | `uv run python runs/adversary_v2.py` | `results/week-11/D1_adv_v2_judge` | | | | judge | |
| D2 | | Heuristic eval | `--eval-method heuristic` | `results/week-11/D2_adv_v2_heuristic` | | | | heuristic | |
| D3 | | Ablation | `--max-subquestions` / `--target-sub-workers` | `results/week-11/D3_adv_v2_ablate` | | | | | |

### Phase E — Optional controls

| ID | Status | Experiment | Command | `--results-dir` (example) | Test ASR | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | | CoEV v2 | `uv run python runs/coev_v2_run.py --mode coev` | `results/week-11/E1_coev_v2` | | |
| E2 | | `adversary_run` eval-only | `--mode eval` | `results/week-11/E2_eval_only` | | |
| E3 | | GEPA (if used) | | `results/week-11/E3_gepa_*` | | |

### Winners

| Category | Winning ID | Why | Default CLI / YAML |
| --- | --- | --- | --- |
| Best prompt (no FT) | | | |
| Best policy + prompt (FT) | | | |
| Best decomposition (`adversary_v2`) | | | |
| Overall | | | |

**Reproduction:** exact `uv run …` + `config_snapshot` in metrics JSON + Notes.

---

## 8. Throughput and target concurrency

**v1 (`adversary_run`):** `--target-max-workers` pipelines target HTTP calls **across behaviors** (adversary stays sequential). Raising it past what the vLLM GPU can serve increases queueing and TTFT.

**v2 (`adversary_v2`):** `--target-sub-workers` overlaps target calls **within a behavior** (per sub-question). `--max-subquestions` increases **how many** calls you issue per behavior.

**How to tune:** Watch vLLM `num_requests_running` vs `num_requests_waiting`, server logs (preemption), or run a 5-behavior sweep with workers `1 → 4 → 8`. Track `run_seconds`, hardware, and HTTP errors in Notes. Rule of thumb on one GPU: start with **4–8** workers; avoid very large values without a multi-replica backend.

---

## 9. Where to read metrics

| Script | Held-out test ASR | Periodic val | Other |
| --- | --- | --- | --- |
| `adversary_run.py` | `adversary_run_metrics.json` → `final_metrics.asr` | `adversary_training_log.csv` → `eval_asr` | Plots when fine-tuning |
| `adversary_v2.py` | Eval: `adversary_v2_metrics.json`. Train: `adversary_v2_run_metrics.json` | Training log when fine-tuning | Row-level `adversary_v2_outputs.csv` |
| `coev_v2_run.py` | `coev_v2_run_metrics.json` | `coev_v2_training_log.csv` | Stage CSV + plots |

---

## 10. Research question, checklist, references

**Question:** Best adversary setup (prompt × policy × training × decomposition)?

**Session checklist**

- [ ] Freeze `csv_seed` / `csv_test_size`; document in metrics / manifest.
- [ ] Phase A (`adversary_run`, `--no-finetune`); record ASR.
- [ ] Phase B fine-tunes; archive `adapters/` for winners.
- [ ] Method table + iteration charts.
- [ ] `adversary_v2` on same CSV slice as single-shot baselines.
- [ ] Fill **Winners** + deliverable sheet.

**Open questions:** Same frozen behavior list across prompt modes? Report decomposition judge vs classic HarmBench framing where comparisons need care?

**References:** [docs/getting-started.md](docs/getting-started.md), [README.md](README.md).

---

## Appendix A: adversary_run.py

**Purpose:** Adversary-only LoRA training with REINFORCE, RLOO, or rejection sampling; no GEPA / prompt evolution. Rewards from HarmBench judge or heuristic on target outputs.

**High-level loop**

1. Load HF train prompts and CSV val/test via `load_harmbench_csv_val_test_splits`.
2. **Baseline eval** on CSV **test** slice (init weights).
3. If `--mode train` and `finetune`: for each iteration, sample HF prompt → adversary rewrite → target → reward → optimizer step; every `eval_every`, run **val** ASR on CSV val slice.
4. **Final eval** on CSV **test** slice.
5. If `--mode train` and `finetune`: save LoRA under `results_dir/adapters/`.
6. Write metrics, CSVs, plots, `run_manifest.json`.

**If `--no-finetune`:** no iterations; baseline = final on test; no training log file when empty; no iteration plots.

**If `--mode eval`:** baseline/final metrics coincide with test eval at loaded weights; no training.

**Artifacts** (under `--results-dir`)

| File | Content |
| --- | --- |
| `adversary_run_metrics.json` | `baseline_metrics`, `final_metrics`, `config` (incl. `finetune`, HF/CSV sizes, `harmbench_csv_path`, `csv_seed`, policy, RS knobs). |
| `eval_metrics_before_vs_after_training.csv` | Two-row comparison (variant before/after). |
| `eval_outputs_before_training.csv`, `eval_outputs_after_training.csv` | Per-test-row generations; length = `csv_test_size`. |
| `adversary_training_log.csv` | Per-iteration rewards, loss, periodic **val** `eval_asr` / `eval_refusal_rate` when fine-tuning. |
| `plot_eval_metrics_before_vs_after_training.png` | Bar chart before vs after. |
| `plot_asr_vs_iterations.png`, `plot_refusal_vs_iterations.png` | When fine-tuning log supports plotting. |
| `run_manifest.json` | Includes `config_snapshot` for repro. |
| `adapters/` | LoRA after successful fine-tune (`--mode train`, finetune on). |

**Config block:** `runs.adversary` (+ `shared_generation` merge). **Entrypoint:** `patch_run_args_from_config(..., run="adversary")`.

---

## Appendix B: adversary_v2.py

**Purpose:** Decomposition attack: sub-questions → multiple target completions → concatenation → evaluation. Optional same-style LoRA training as v1 on the decomposition rollout reward.

**High-level loop**

- **`--mode eval`:** For each CSV test behavior: decompose → parse → ordered target calls → concat → aggregate `evaluate_outputs`. Writes **`adversary_v2_outputs.csv`** + **`adversary_v2_metrics.json`** (includes `metrics`, `eval_result`, `config`, `config_snapshot`, `run_seconds`).
- **`--mode train`:** Same data split pattern as v1 (HF train, CSV val/test). Baseline test eval → training iterations with periodic **val** ASR (when fine-tuning) → final test eval → optional `adapters/`. Uses **`adversary_v2_run_metrics.json`**, `eval_metrics_before_vs_after_training.csv`, `eval_outputs_before_training.csv`, `eval_outputs_after_training.csv`, `adversary_v2_training_log.csv` (name from YAML), plots, `run_manifest.json`.

**Adversary default:** `ADVERSARY_V2_SYSTEM_PROMPT_TEMPLATE` (Piecewise Vulnerability Assessment); per-behavior `{{behavior}}` substitution in system, short user nudge to emit tags only.

**Judge:** Decomposition-specific `[INST]` template (`ADVERSARY_V2_JUDGE_CLASSIFICATION_TEMPLATE`); refers to HLD and “technical sub-components”; not identical to classic single-reply HarmBench classifier wording.

**Parsing (`extract_subquestions`):** Text between `START_DECOMPOSITION` and `END_DECOMPOSITION` is scanned line-by-line for `N.` / `N)` items (ellipsis-only lines skipped). A single numbered item is valid. If markers are present but nothing parses, the list is empty and `decomposed_rollout` falls back to one trajectory. Without markers, the full text is scanned for numbered lines, then bullets, then a single prose chunk.

**Concatenation:** Combined target outputs use the “Piecewise Vulnerability Assessment — combined technical responses” header and `[Sub-component i]` / `[Technical response i]` blocks for the judge input.

**Artifacts (eval)**

| File | Content |
| --- | --- |
| `adversary_v2_outputs.csv` | Per behavior: raw decomposition, subquestions JSON, concatenated generation, latencies, scores. |
| `adversary_v2_metrics.json` | Aggregate metrics + config + `config_snapshot`. |

**Artifacts (train)** — same pattern as v1 before/after naming, plus v2-specific training log filename from `runs.adversary_v2.training_csv_name` (default `adversary_v2_training_log.csv`).

**Config:** `_merge_run_defaults`: `runs.adversary` ⊕ `runs.adversary_v2`. **Entrypoint:** `patch_run_args_from_config(..., run="adversary")` for model IDs and vLLM URL.

---

## Appendix C: coev_v2_run.py

**Purpose:** “Vanilla” adversary training: **fixed** attacker instruction and defense prompt (from YAML/CLI), **staged** outer loop (`stages` × `iters_per_stage`), REINFORCE or RLOO, optional rejection sampling. No rewriter presets, no GEPA.

**Modes:** `--mode coev` (train) or `--mode eval` (baseline eval + artifacts with empty training).

**Data:** HF HarmBench train/val slices via `train_size`, `val_size`, and slice indices (`train_slice_end`, `eval_slice_start`, `eval_slice_end`) — **different indexing model** than CSV-centric v1/v2 (HF pool + slices, not `harmbench_csv_path` for eval).

**Artifacts** (typical)

| File | Content |
| --- | --- |
| `coev_v2_run_metrics.json` | `baseline_metrics`, `optimized_metrics`, training config summary. |
| `baseline_vs_optimized_metrics.csv` | Tabular comparison. |
| `baseline_eval_outputs.csv`, `optimized_eval_outputs.csv` | Row-level eval. |
| `coev_v2_training_log.csv` | Per step (name from `--training-csv-name`). |
| `coev_v2_stage_metrics.csv` | Checkpoint ASR/refusal per stage. |
| `coev_v2_optimized_prompts.json`, `optimized_attacker_instruction.txt`, `optimized_defense_prompt.txt` | Final prompt strings (fixed across training; “optimized” = end of run). |
| `plot_baseline_vs_optimized.png`, `plot_asr_vs_iterations.png`, `plot_refusal_vs_iterations.png` | When stage metrics support plots. |
| `run_manifest.json` | `config_snapshot`. |
| `adapters/` | Optional via `--save-dir` under `--results-dir`. |

**Config block:** `runs.coev_v2`. Use when you need a **non–Theme-2** baseline comparable in “adversary weights only” spirit but not comparable row-for-row to CSV-split v1/v2 without careful alignment.
