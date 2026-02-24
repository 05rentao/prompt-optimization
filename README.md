## Prompt Optimization and Adversarial Attacks on GEPA‑Aided Models

This repository contains the code and experiments for a STAT 4830 course project on **prompt optimization, GEPA, and adversarial attacks on aligned language models**. Conceptually, the project has two main threads:

- **GEPA baselines**: Notebooks (`GEPA_baselines.ipynb`, `Stat4830_project.ipynb`) that use GEPA to generate robust system prompts and evaluate toxicity/robustness of LLMs.
- **Jailbreak + steering experiments**: Python modules under `src/` that load a small instruction‑tuned model (Qwen), run a **jailbreak evaluation loop** on curated attack data, and experiment with **activation steering** to reduce attack success.

For more background and the full problem statement, see `report.md`.

---

## Project Structure

- **`src/`**: Core Python modules for jailbreak evaluation and steering
  - `main.py`: Example end‑to‑end script that
    - loads a target model and a judge model via `load_model_and_tokenizer`,
    - reads a small jailbreak benchmark from `data/jailbreak_10.csv`,
    - evaluates **baseline attack success** with `JailbreakEvaluator`,
    - then applies an activation‑addition steering vector and re‑evaluates.
  - `evaluator.py`: Defines `JailbreakEvaluator`, which
    - generates model responses to adversarial goals,
    - uses a separate judge model to classify each response as “jailbroken” or “safe” based on logits for `"Yes"` vs `"No"`,
    - returns an aggregate **attack success rate** plus per‑example results.
  - `models.py`: Utility for `load_model_and_tokenizer` using Hugging Face `AutoModelForCausalLM` and `AutoTokenizer` with sensible dtypes and device mapping.
  - `steering_methods/base.py`: Abstract base class `SteeringMethod` providing a standard `apply(...)` interface and `remove()` hook cleanup for steering methods.
  - `steering_methods/activation_add.py`: Implements **activation addition steering** via `ActAddSteering`, attaching a forward hook to a specified Qwen transformer layer and adding a learned steering vector to the hidden states.
  - `steering_methods/sae.py`: Placeholder / debugging file (currently just imports `torch` and prints a range).

- **`data/`**: Jailbreak and steering‑related datasets
  - `jailbreak_10.csv`: Small CSV benchmark of jailbreak prompts with columns like `goal`, `behavior`, `category`, and `source` (used by `main.py` and `JailbreakEvaluator`).
  - `jailbreak_100.json`: Larger JSON version of the jailbreak dataset (can be converted to CSV with `scripts/json_to_csv.py`).
  - `activation_vector_refusal.json`: Saved activation/steering information for refusal behavior (e.g., for building steering vectors).

- **`scripts/`**: Utility scripts for data prep and environment testing
  - `json_to_csv.py`: Converts a nested JSON jailbreak dataset into a flat CSV.
  - `gepa-test.py`, `single_step_test.py`, `cpu test.py`, `gpu version.py`: Miscellaneous testing scripts (e.g., environment checks, GEPA or device tests).

- **Notebooks**
  - `GEPA_baselines.ipynb`: Runs GEPA‑based baselines, evaluates toxicity, and explores different teacher/student model choices.
  - `Stat4830_project.ipynb`: Higher‑level project notebook with early experiments and documentation.

- **Other files**
  - `report.md`: Detailed write‑up of the project problem, optimization framing, and preliminary GEPA results.
  - `requirements.txt`: Full Python dependency list used for the experiments.
  - `.gitignore`: Git ignore rules for the repo.

---

## Environment Setup

You can either create a new **Conda** environment or activate an existing one.

- **Create a new environment:**

```bash
conda create -n prompt-optimization python=3.12.12 
conda activate prompt-optimization
```

- **Or activate an existing one:**

```bash
conda activate your-env-name
```

Once your environment is active, install the project requirements:

- **Using `uv` (recommended, optional):**

```bash
uv pip install -r requirements.txt
```

- **Using standard `pip`:**

```bash
pip install -r requirements.txt
```

To confirm that packages were installed correctly:

```bash
pip list
```

> **Tip:** If you add new libraries to the project, update the requirements file using `pip freeze > requirements.txt`.

---

## Running the Jailbreak + Steering Experiment

- **1. Prepare data**
  - Ensure `data/jailbreak_10.csv` exists. If you have `jailbreak_100.json` instead, you can adapt `scripts/json_to_csv.py` to export a CSV with columns
    `index, goal, target, behavior, category, source`.
  - (Optional) Place precomputed steering vectors under a `vectors/` directory, e.g. `vectors/refusal_vector_layer15.pt`.

- **2. Configure models**
  - In `src/main.py`, the default configuration is:
    - Target model: `"Qwen/Qwen2.5-1.5B-Instruct"`
    - Judge model: `"Qwen/Qwen2.5-1.5B-Instruct"`  
  - Both are loaded through `load_model_and_tokenizer` in `models.py`. You can swap in any compatible Hugging Face causal LM by changing the model names.

- **3. Run the script**

From the repository root:

```bash
python -m src.main
```

This will:

- load the target and judge models,
- compute a baseline **attack success rate (ASR)** on `data/jailbreak_10.csv` using `JailbreakEvaluator.run_eval(...)`,
- load a refusal steering vector (e.g. `vectors/refusal_vector_layer15.pt`),
- attach an activation‑addition hook to a chosen Qwen layer via `ActAddSteering.apply(...)`,
- run the evaluation again under steering,
- print both the baseline and steered ASR to the console.

You can adjust:

- `layer_idx` (which transformer block to steer),
- `coefficient` (how strongly to apply the steering vector),
- the steering vector file path,
- and the dataset path.

---

## GEPA Baselines and Project Context

Although the Python modules under `src/` focus on Qwen‑based jailbreak/steering, the overall project centers on **GEPA‑protected systems** and **adversarial prompt optimization**. The typical workflow is:

- Use GEPA (see `GEPA_baselines.ipynb`) to learn a robust system prompt for a student model from a dataset of prompts and answers.
- Evaluate toxicity and safety metrics (e.g., via Detoxify) under different student/teacher pairs.
- Independently, use the `src/` code to study **jailbreak attack rates** and **activation‑based defenses**, which complements the GEPA studies by exploring a different part of the safety toolbox (representation steering instead of system‑prompt engineering).

For the full optimization framing, metrics, and discussion of limitations, see `report.md`.

---

## Reproducibility and Notes

- **Hardware**: Experiments assume access to a GPU with sufficient VRAM (e.g., 16–24GB) due to Qwen model sizes and token lengths.
- **Randomness**: Generation uses sampling (`do_sample=True`, `temperature=0.7`); for fully reproducible results, set a global random seed for PyTorch, NumPy, and the transformers library.
- **Extensibility**: New steering methods can be added by subclassing `SteeringMethod` under `src/steering_methods/` and implementing an `apply(...)` method that attaches/removes hooks as needed.