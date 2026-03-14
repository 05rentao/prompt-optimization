# `mark_exp.py` Logic and Pipeline Guide

This document explains the converted script in `experimental_code_2/mark_exp.py`, including what it does, how the core pipeline works, and where to refactor next.

## Purpose

The script runs an end-to-end GEPA prompt optimization workflow for safety behavior:

- Uses a harmful-request subset from HarmBench.
- Evaluates a baseline system prompt against refusal-oriented metrics.
- Optimizes the prompt with GEPA using a reflection model.
- Re-evaluates the optimized prompt on the same validation subset.
- Exports metrics, model outputs, and visual artifacts.

## Inputs and Runtime Dependencies

### External services

- **Task model endpoint**: local vLLM server exposed as OpenAI-compatible API (`--vllm-base-url`).
- **Reflection model endpoint**: OpenAI API (`--openai-api-key`, `--reflection-model-name`).
- **Dataset source**: Hugging Face dataset (`--dataset-name`, `--dataset-config`, `--dataset-split`).

### Python packages

Core packages used by the script:

- `gepa`
- `openai`
- `datasets`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tqdm`

## Core Pipeline

```mermaid
flowchart TD
    parseArgs[ParseArgsAndEnv] --> maybePrompt[PromptMissingSecretsOptional]
    maybePrompt --> loadData[LoadAndNormalizeHarmBenchSubset]
    loadData --> verifyClients[VerifyTaskAndReflectionClients]
    verifyClients --> baselineEval[EvaluateBaselineSystemPrompt]
    baselineEval --> runOpt[RunGepaOptimization]
    runOpt --> extractBest[ExtractBestCandidateAndScore]
    extractBest --> optimizedEval[EvaluateOptimizedSystemPrompt]
    optimizedEval --> export[SaveMetricsCsvPlotsPrompt]
```

## Step-by-Step Functionality

1. **Argument and environment parsing**
   - `parse_args()` defines all tunable parameters (dataset sizes, model names, endpoint URLs, token limits, optimization budget).
   - API keys and tokens default to environment variables but can be passed by CLI.

2. **Optional hidden credential prompt**
   - `maybe_prompt_missing_secrets()` allows runtime credential entry when `--prompt-for-missing-secrets` is set.
   - Useful when notebook/kernel-like environments do not inherit shell exports.

3. **Dataset loading and normalization**
   - `load_harmbench_subset()` loads a gated or public HF dataset split, shuffles deterministically, and slices train/val subsets.
   - `normalize_record()` auto-detects prompt/id fields from candidate key lists and standardizes records.
   - Output records use a consistent schema: `id`, `prompt`, `is_harmful_request=True`.

4. **Task/reflection connectivity checks**
   - `verify_task_and_reflection_clients()`:
     - validates vLLM endpoint reachability,
     - confirms requested task model is served,
     - runs a short smoke generation for task model,
     - verifies reflection model access via OpenAI API.

5. **Metric definition and baseline evaluation**
   - `refusal_score()` applies a simple refusal-phrase heuristic.
   - `evaluate_system_prompt()` runs inference per example and computes:
     - `refusal_rate`
     - `unsafe_compliance_rate`
     - `mean_refusal_score`
     - `aggregate_score`
     - latency and throughput proxies
   - Baseline prompt is evaluated first on validation data.

6. **GEPA optimization loop**
   - `run_gepa_optimization()` imports GEPA runtime and builds:
     - train/validation datasets in GEPA format,
     - evaluator callback for candidate prompt scoring,
     - `GEPAConfig` with `EngineConfig(max_metric_calls=...)`,
     - `ReflectionConfig(reflection_lm="openai/<model>")`.
   - Evaluator returns score + side info and appends per-call trace records.

7. **Best prompt extraction and re-evaluation**
   - `extract_best_candidate_and_score()` handles both dict and object result styles from GEPA.
   - Optimized prompt falls back to baseline if missing.
   - Optimized prompt is evaluated with the same validation protocol.

8. **Artifact export**
   - `save_artifacts()` writes:
     - `optimized_system_prompt.txt`
     - `gepa_run_metrics.json`
     - `results/baseline_vs_optimized_metrics.csv`
     - `results/baseline_eval_outputs.csv`
     - `results/optimized_eval_outputs.csv`
     - `results/optimizer_trace.csv` (if trace exists)
     - `results/plot_baseline_vs_optimized.png`
     - `results/plot_optimization_trajectory.png` (if trace exists)

## CLI Usage

Example command:

```bash
python experimental_code_2/mark_exp.py \
  --task-model-name "Qwen/Qwen2.5-3B-Instruct" \
  --reflection-model-name "gpt-4o-mini" \
  --dataset-name "walledai/HarmBench" \
  --dataset-config "standard" \
  --dataset-split "train" \
  --train-size 100 \
  --val-size 100 \
  --max-metric-calls 300 \
  --show-progress
```

If secrets are not exported:

```bash
python experimental_code_2/mark_exp.py --prompt-for-missing-secrets
```

## Output Interpretation

- Improvement target is higher `aggregate_score` and `refusal_rate`, lower `unsafe_compliance_rate`.
- Because scoring is heuristic, manual spot checks of generated responses are recommended.
- Runtime metrics (`latency_ms_mean`, `tokens_per_second_proxy`) are informative but hardware/load dependent.

## Refactoring Opportunities (Based on `mark_exp.py`)

### 1) Split into modules by concern

Current script is cohesive but monolithic. Break into:

- `config.py` for CLI/env/dataclasses
- `data.py` for dataset loading and normalization
- `evaluation.py` for scoring and metrics
- `optimization.py` for GEPA integration
- `reporting.py` for plots/artifact exports

This improves readability, testability, and reuse.

### 2) Introduce typed config/result dataclasses

Replace loose `argparse.Namespace` and nested dict payloads with dataclasses such as:

- `RunConfig`
- `DatasetConfig`
- `ModelConfig`
- `MetricsSummary`
- `OptimizationResultSummary`

Benefits: clearer contracts and reduced key/field drift.

### 3) Replace heuristic scorer with pluggable evaluators

`refusal_score()` is lightweight and transparent but brittle. Introduce an evaluator interface:

- `HeuristicRefusalEvaluator`
- `ClassifierBasedSafetyEvaluator`
- `LlmJudgeSafetyEvaluator`

Then compose aggregate metrics from configurable evaluator weights.

### 4) Add robust retry/backoff and failure isolation

Network calls are central and should be hardened:

- retry policy for transient API failures/timeouts
- per-example failure capture without aborting whole run
- fail-fast mode toggle vs best-effort mode

### 5) Add caching/checkpointing

Long runs can be expensive. Add:

- response cache keyed by `(system_prompt, input, model_name, gen_params)`
- periodic checkpointing of optimizer trace and intermediate best candidate
- resume option for interrupted runs

### 6) Improve experiment reproducibility and tracking

Introduce run IDs and metadata logging:

- immutable run directory like `results/<timestamp_or_uuid>/`
- capture CLI args, git SHA, package versions, and environment summary
- optional MLflow/W&B integration for comparing runs

### 7) Add test seams and unit/integration tests

Testing targets:

- `normalize_record()` schema edge cases
- `refusal_score()` deterministic behavior
- `extract_best_candidate_and_score()` compatibility cases
- mocked OpenAI client for `evaluate_system_prompt()`
- smoke integration test with tiny sample sizes and minimal budget

## Suggested Near-Term Refactor Order

1. Introduce dataclasses and typed config objects.
2. Extract evaluation/scoring into `evaluation.py`.
3. Extract GEPA runner into `optimization.py`.
4. Add network retry + per-example error handling.
5. Add checkpointing and run-directory versioning.
