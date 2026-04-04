# Adversary Optimization (Single-Notebook Implementation)

This project provides a notebook-first implementation for adversary-model optimization in a controlled safety research setting.

## Project Structure

- `notebooks/main.ipynb`: Primary implementation notebook (all core logic).
- `literature_review.md`: Literature context and methodology reference.
- `requirements.txt`: Reproducible dependency list.
- `.env.example`: Environment variable template.
- `outputs/`: Generated artifacts from notebook runs.

## Scope

- The notebook implements a complete pipeline:
  - Data loading and stratified split.
  - PAIR-style attack rewriting strategies.
  - Task-model interaction loop.
  - Evaluator scoring and metric computation.
  - RL-style strategy optimization (single-notebook baseline).
  - Ablation and artifact export.
- Optional PPO upgrade is included as a scaffold for future extension.

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

## Backend Modes

Set backend modes in the notebook config cell:

- `attacker_backend`: `mock` | `hf` | `openai`
- `task_backend`: `mock` | `hf` | `openai`
- `evaluator_backend`: `mock` | `openai`

Recommended for first run:
- `mock` / `mock` / `mock` to validate pipeline execution quickly.

## Outputs

Each run writes a timestamped output directory under `outputs/` containing:

- `train_logs.csv`
- `val_logs.csv`
- `test_logs.csv`
- `summary.json`
- `ablation.csv`
- `strategy_state.json`
- `asr_curve.png`

## Troubleshooting

- If package imports fail, reinstall requirements and restart kernel.
- If Hugging Face model loading fails, verify model access and local GPU memory.
- If OpenAI evaluation fails, verify `OPENAI_API_KEY` and API quota.
- If HarmBench dataset loading fails, the notebook uses a synthetic fallback dataset so the pipeline can still run.

## Research Safety Notes

- Use only in authorized safety evaluation contexts.
- Do not test against systems without explicit permission.
- Keep datasets and generated artifacts in controlled storage.
