# Run artifacts reference

Quick reference for what each runner writes under `results_dir`. Use this to compare runs, debug GEPA, or wire dashboards.

**`run_manifest.json`** (all runners below) includes a **`config_snapshot`** object: which YAML was loaded (`PROMPT_OPT_CONFIG_PATH` or `configs/default.yaml`), effective reflection base URL and related env flags (no API key values), and **`cli_args`** — the resolved argparse values (defaults from YAML are already applied where the script wires them). The full YAML is **not** embedded.

## Summary

| Runner | Script | Trains weights | GEPA | Typical use |
|--------|--------|----------------|------|-------------|
| Adversary-only | `runs/adversary_run.py` | Yes | No | REINFORCE on adversary; fixed prompts |
| GEPA-only | `runs/gepa_run.py` | No | Yes (1 role: system prompt) | Isolate defense prompt optimization |
| CoEV v2 | `runs/coev_v2_run.py` | Yes (REINFORCE) | Yes (2 roles / stage) | Full coevolution |
| CoEV v2 RLOO | `runs/coev_v2_RLOO_run.py` | Yes (RLOO) | Same as CoEV v2 | Same artifacts as CoEV v2; training CSV differs |

CoEV v2 and RLOO share the **same filenames**; distinguish runs via `run_manifest.json` (`mode`: `coev_v2` vs `coev_v2_rloo`) and plot titles.

---

## Adversary-only (`adversary_run.py`)

**What is being compared?** This run **does not** optimize prompts. The rewriter instruction and target system prompt stay fixed (from CLI / `configs/default.yaml`). The **adversary model’s LoRA weights** are trained with REINFORCE. Artifacts compare **eval on the held-out set** at **initialization** vs **after** training.

| Artifact | Purpose |
|----------|---------|
| `adversary_run_metrics.json` | `baseline_metrics` (before any training step) and `final_metrics` (after training, or same as baseline in eval-only mode), plus run config |
| `eval_metrics_before_vs_after_training.csv` | Two rows: `before_training` vs `after_training` aggregate metrics (same meaning as the bar chart) |
| `eval_outputs_before_training.csv`, `eval_outputs_after_training.csv` | Per-example adversary→target eval for each phase |
| `<training_csv_name>` (from config) | Per-iteration training log; skipped if empty (eval-only) |
| `plot_eval_metrics_before_vs_after_training.png` | Bar chart of aggregate metrics; title/subtitle spell out before/after **weights** |
| `plot_asr_vs_iterations.png` | Periodic eval **attack success rate (ASR)** vs iteration (when training log has `eval_asr`) |
| `plot_refusal_vs_iterations.png` | Same checkpoints as **refusal rate** (`1 - ASR`); clearer for “target still refuses” |
| `run_manifest.json` | `mode`: `adversary_train` / `adversary_eval`, models, budget, `endpoints.reflection_base_url` (vLLM URL for target) |

**Optional:** `--save-dir` (adversary LoRA checkpoints): relative paths are resolved under **`--results-dir`**, not the repo root.

**GEPA:** does not run; no reflection/GEPA health here.

---

## GEPA-only (`gepa_run.py`)

| Artifact | Purpose |
|----------|---------|
| `optimized_system_prompt.txt` | Best system prompt string |
| `gepa_run_metrics.json` | `baseline_metrics`, `optimized_metrics`, **`best_score_from_gepa`**, reflection URL, `max_metric_calls` |
| `baseline_vs_optimized_metrics.csv`, `baseline_eval_outputs.csv`, `optimized_eval_outputs.csv` | Direct user→target eval (no adversary) |
| `optimizer_trace.csv` | One row per evaluator call during GEPA |
| `plot_baseline_vs_optimized.png`, `plot_optimization_trajectory.png`, `plot_refusal_vs_evaluator_calls.png` | Bar + score trajectory + per-call refusal vs cumulative best refusal |
| `run_manifest.json` | `mode`: `gepa` |

All paths are under **`--results-dir`** (default from `runs.gepa.results_dir` in config).

**GEPA sanity checks:** non-empty `optimizer_trace.csv`, `best_score_from_gepa` in JSON/manifest, prompt file updated vs baseline, trajectory plot shows variation when the objective is non-flat.

---

## CoEV v2 & CoEV v2 RLOO (`coev_v2_run.py`, `coev_v2_RLOO_run.py`)

| Artifact | Purpose |
|----------|---------|
| `coev_v2_optimized_prompts.json` | Final `attacker_instruction` + `defense_prompt` |
| `optimized_attacker_instruction.txt`, `optimized_defense_prompt.txt` | Same, plain text |
| `coev_v2_run_metrics.json` | `baseline_metrics`, `optimized_metrics`, `config` (stages, `max_metric_calls`, reflection URL, …), optional **`gepa_best_val_scores_final`** `{attacker, defender}` after training |
| `baseline_vs_optimized_metrics.csv` | Full-run baseline vs final eval |
| `baseline_eval_outputs.csv`, `optimized_eval_outputs.csv` | Per-example adversary→target eval |
| `<training_csv_name>` (config) | REINFORCE vs RLOO columns depending on script |
| `coev_v2_stage_metrics.csv` | Per **stage**: `pre_evolution`, `attacker_gepa_best`, `defender_gepa_best`, `gepa_seconds` (+ eval metrics on GEPA rows when enabled) |
| `optimizer_trace_attacker.csv`, `optimizer_trace_defender.csv` | GEPA evaluator traces (two `optimize_anything` runs per stage) |
| `plot_baseline_vs_optimized.png`, `plot_optimization_trajectory.png` | Bar chart + GEPA trajectory (per-role call index; per-role cumulative best) |
| `plot_asr_vs_iterations.png` | Checkpoint eval **ASR** vs synthetic global step (pre-evolution + post-GEPA offsets) |
| `plot_refusal_vs_iterations.png` | Same checkpoints as **refusal rate** (defense-aligned) |
| `run_manifest.json` | `mode`: `coev_v2` or `coev_v2_rloo`, models, budget, `endpoints.reflection_base_url` |

**Optional:** `--save-dir` (adversary LoRA): relative paths resolve under **`--results-dir`**.

**GEPA sanity checks:** `coev_v2_stage_metrics.csv` has GEPA phases and positive `gepa_seconds`; traces non-empty; `gepa_best_val_scores_final` when training ran; startup reflection smoke print succeeds (failures usually abort before save).

---

## Cross-run comparisons

- **Shared names (good for tooling):** `baseline_vs_optimized_metrics.csv`, `baseline_eval_outputs.csv`, `optimized_eval_outputs.csv`, `plot_baseline_vs_optimized.png`, `run_manifest.json` (GEPA / CoEV v2). Adversary-only uses the `eval_*` filenames above instead.
- **Merge candidates (optional):** concatenate `optimizer_trace_attacker.csv` + `optimizer_trace_defender.csv` with a `role` column (mirrors the trajectory plot); not required for correctness.

---

## Artifact usefulness (short)

| Kind | Use |
|------|-----|
| `*_metrics.json` | Repro, config audit, comparing runs |
| Baseline vs optimized CSV + bar plot | Did metrics move in the expected direction? |
| Per-example eval CSVs | Debugging examples, judge vs heuristic, latency |
| Prompt JSON + `.txt` | Reuse prompts elsewhere |
| Optimizer trace CSVs | Proof GEPA ran; call counts and rough progression |
| `coev_v2_stage_metrics.csv` | **Best** staged view for CoEV GEPA |
| Trajectory plot | Visual GEPA behavior (noisy objectives) |
| `run_manifest.json` | Index of paths and hyperparameters |
| Training CSV | RL debugging; not GEPA-specific |
