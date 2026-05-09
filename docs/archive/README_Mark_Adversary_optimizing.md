# Adversary Optimization (Notebook-First)

This project provides a notebook-first workflow for adversary-model optimization in a controlled safety research setting.

## Project Structure

- `notebooks/main.ipynb`: Primary implementation notebook (all core logic).
- `notebooks/experiment_matrix.ipynb`: Control-group and parameter-sweep notebook for comprehensive comparisons.
- `literature_review.md`: Literature context and methodology reference.
- `requirements.txt`: Reproducible dependency list.
- `.env.example`: Environment variable template.
- `outputs/`: Generated artifacts from notebook runs.

## Notebook Roles

- `notebooks/main.ipynb`
  - Baseline end-to-end pipeline.
  - Data loading and stratified split.
  - PAIR-style rewriting + RL-style strategy updates.
  - Evaluator metrics, visualization dashboard, and artifact export.
- `notebooks/experiment_matrix.ipynb`
  - Comprehensive comparison notebook.
  - Control groups (no-rewrite, random-rewrite, pair-only, success-only).
  - Parameter sweeps (epochs, strategy LR, temperature, max tokens, data scale).
  - Multi-seed reproducibility checks.
  - Matrix summary visualization and exports.

Optional PPO upgrade remains a scaffold for future extension.

## Environment Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment template (optional for API backends):

```bash
copy .env.example .env
```

4. If using OpenAI backends, set values in `.env`:
- `OPENAI_API_KEY`
- Optional model overrides.

## Run in VS Code Notebook

1. Open `notebooks/main.ipynb`.
2. Select a Python kernel with installed dependencies.
3. Run cells from top to bottom.
4. Keep markdown notes and code comments in English.

### Recommended Order

1. Run `notebooks/main.ipynb` to validate baseline functionality and output artifacts.
2. Run `notebooks/experiment_matrix.ipynb` to execute control/sweep/reproducibility experiments.

### Quick Start Modes

- Fast smoke test:
  - Use `mock` / `mock` / `mock` backends.
  - In `experiment_matrix.ipynb`, set `max_runs = 5`.
- Full experiment matrix:
  - Keep selected backend configuration.
  - In `experiment_matrix.ipynb`, set `max_runs = None`.

## Backend Modes

Set backend modes in the notebook config cell:

- `attacker_backend`: `mock` | `hf` | `openai`
- `task_backend`: `mock` | `hf` | `openai`
- `evaluator_backend`: `mock` | `openai`

Recommended for first run:
- `mock` / `mock` / `mock` to validate pipeline execution quickly.

## Outputs

Each run writes a timestamped output directory under `outputs/` containing:

From `main.ipynb`:
- `train_logs.csv`
- `val_logs.csv`
- `test_logs.csv`
- `summary.json`
- `ablation.csv`
- `strategy_state.json`
- `asr_curve.png`
- `category_asr.png`
- `dashboard.png`

From `experiment_matrix.ipynb`:
- `matrix_summary.csv`
- `matrix_summary.json`
- `matrix_overview.png`

## Troubleshooting

- If package imports fail, reinstall requirements and restart kernel.
- If Hugging Face model loading fails, verify model access and local GPU memory.
- If OpenAI evaluation fails, verify `OPENAI_API_KEY` and API quota.
- If HarmBench dataset loading fails, the notebook uses a synthetic fallback dataset so the pipeline can still run.
- If matrix execution is slow, reduce `max_runs` or lower `max_behaviors` in config.

## Research Safety Notes

- Use only in authorized safety evaluation contexts.
- Do not test against systems without explicit permission.
- Keep datasets and generated artifacts in controlled storage.
