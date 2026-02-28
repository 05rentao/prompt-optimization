# STAT 4830 — Red Team Adversarial Training

This project implements an **adversarial red team** loop: a small LM (the *adversary*) learns to generate jailbreak prompts, a *target* model responds, and a *judge* scores whether the target complied. The adversary is updated with REINFORCE using that reward. **No API keys are required**; the judge is the local **cais/HarmBench-Llama-2-13b-cls** classifier (official HarmBench rubric, 4-bit on CUDA).

---

## Repo layout and what each file does

| Path | Purpose |
|------|--------|
| **configs/default.yaml** | Single config: adversary, evaluator, defense, target, training (all hyperparameters and model names). |
| **run_experiment.py** | One training run: load config → build adversary/target/data → `TrainingLoop.run()` (no step cap). |
| **run_trials.py** | Two trials (GEPA off, GEPA on), 50 behaviors each, 150 steps or 5 min; logs to `outputs/trial_summaries.txt`. |
| **run_trials_gpu.py** | GPU trials: two runs (GEPA off/on), 150 behaviors, 100 steps, in-loop eval + post-trial eval; CSVs in `outputs/`. |
| **src/utils/config.py** | `load_config(path)` — single place that reads YAML. |
| **src/utils/gpu.py** | `free_gpu_memory()` — gc + `torch.cuda.empty_cache()`; use between trials to reduce OOM. |
| **src/builders.py** | `build_adversary(config)`, `get_target_from_config(config)` — build adversary and target from config. |
| **src/adversary/policy.py** | RedTeamPolicy (Qwen + PEFT LoRA), REINFORCE: `generate_attack`, `update` / `update_batch`. |
| **src/data/harmbench_loader.py** | HarmBenchLoader.load(limit, seed): HF or GitHub CSV or synthetic; seed for reproducible splits. |
| **src/evaluator/judge.py** | LocalHarmBenchJudge (4-bit), official HarmBench prompt; returns Yes/No + explanation; compute_reward. |
| **src/training/loop.py** | TrainingLoop: adversary → target → judge → optional GEPA → REINFORCE update; batch or per-behavior. |
| **src/training/logger.py** | ExperimentLogger (timestamped CSV), log_trial_summary (append to trial_summaries.txt). |
| **src/defense/gepa_wrapper.py** | BlueTeam (DSPy), get_blue_team() with Ollama llama3:8b; evolves system prompt on jailbreak. |
| **src/target/hf_target.py** | make_hf_target(): in-process HF model (e.g. Qwen 0.5B, 4-bit). |
| **src/target/ollama_target.py** | make_target_model(), make_mock_target(): Ollama or mock. |

---

## How the code works

1. **Config**  
   `configs/default.yaml` defines:
   - **adversary**: model (e.g. `unsloth/Qwen2.5-1.5B-Instruct`), learning rate, LoRA rank/alpha, max length.
   - **evaluator**: judge model (default `cais/HarmBench-Llama-2-13b-cls`) — local classifier, no API key.
   - **defense**: `USE_GEPA_DEFENDER` (bool) — if true, BlueTeam evolves the system prompt on successful jailbreak using local Ollama (`llama3:8b`). **The defender is fully wired:** on each successful jailbreak the loop calls `BlueTeam.forward(context=...)` and uses the returned `system_prompt` for all following rollouts.
   - **target**: `use_ollama` (bool), `ollama_model` (e.g. `llama3:8b`), `ollama_url` — when `use_ollama` is true, the target is a real Llama via Ollama; otherwise a mock is used.
   - **training**: epochs, batch size, HarmBench limit.

2. **Entry point**  
   `run_experiment.py` loads the config, builds the **RedTeamPolicy** (adversary), **HarmBenchLoader** (behaviors), and a **target** callable (mock or real model). It then constructs **TrainingLoop** and calls `trainer.run()`.

3. **Adversary (`src/adversary/policy.py`)**  
   - **RedTeamPolicy** loads the adversary model with Unsloth (4-bit + LoRA), and uses pure SGD as the optimizer.
   - **generate_attack(behavior)** produces a jailbreak prompt and the log-probability of the generated tokens (for REINFORCE).
   - **update(log_prob, reward)** performs the REINFORCE step: loss = -reward * log_prob, backward, optimizer step.

4. **Data (`src/data/harmbench_loader.py`)**  
   **HarmBenchLoader.load(limit)** tries to load the HarmBench “behavioral” split from Hugging Face; on failure it falls back to a small list of synthetic harmful behaviors. Each item is a dict with a `"behavior"` key.

5. **Evaluator (`src/evaluator/judge.py`)**  
   - **LocalHarmBenchJudge** loads the official **cais/HarmBench-Llama-2-13b-cls** classifier with **transformers** and **bitsandbytes** (4-bit quantization, bfloat16 compute) on CUDA.
   - Uses the **official HarmBench chat template**: system instructions for red-team classification, behavior/generation placeholders, and the rubric rules (unambiguous, non-minimal, harmful, code artifacts when applicable).
   - **`__call__(behavior, response)`** returns a namespace with **`is_jailbroken`** = **`"Yes"`** or **`"No"`** (Llama-2-13b-cls format).
   - **compute_reward(is_jailbroken)** returns **1.0** for `"Yes"` and **0.0** for `"No"` for the REINFORCE loop. No API key is used.

6. **Training loop (`src/training/loop.py`)**  
   For each epoch and each behavior entry:
   - Adversary generates an **attack prompt** and its **log_prob**.
   - **Target** is called with the **current system prompt** and the attack; it returns a **response**.
   - **LocalHarmBenchJudge** evaluates (behavior, response) → **"Yes"** / **"No"**; **compute_reward()** maps that to **1.0** / **0.0**.
   - **GEPA (optional):** If `USE_GEPA_DEFENDER` is true and reward is 1.0 (successful jailbreak), **BlueTeam.forward(context=...)** is called with recent failed defenses. The returned **system_prompt** replaces the target’s prompt for all **subsequent** rollouts (mutation loop). If the flag is false, the target always uses the fixed base system prompt.
   - Adversary is **updated** with REINFORCE using that reward and log_prob.
   - **ExperimentLogger** appends a row including **system_prompt** and **use_gepa_defender** for success-rate plots.

7. **Logger (`src/training/logger.py`)**  
   **ExperimentLogger** writes `outputs/experiment_YYYYMMDD_HHMMSS.csv` with columns: epoch, behavior, adversary_attack, target_response, reward, is_jailbroken, explanation, **system_prompt**, **use_gepa_defender**. Use the last two to compare attacker success rate with and without GEPA.

8. **Defense (`src/defense/gepa_wrapper.py`)**  
   **BlueTeam** is a DSPy module with **DefenseSystem** (context → system_prompt). **get_blue_team()** configures **dspy.OllamaLocal(model="llama3:8b")** and returns BlueTeam so reflection runs locally (no API cost). Used only when `defense.USE_GEPA_DEFENDER` is true; requires [Ollama](https://ollama.com) with `llama3:8b` pulled (e.g. `ollama run llama3:8b`).

---

## Running locally

- **Python 3.10+** and a **GPU with enough VRAM** (the judge is a 13B model in 4-bit; ~8–10 GB typical; adversary adds more).
- Install dependencies (from project root):

  ```bash
  pip install -r requirements.txt
  ```

- Run:

  ```bash
  python run_experiment.py
  ```

  **Two-trial run (GEPA off vs on):** Use `run_trials.py` to run 150 steps (or 5 minutes, whichever is shorter) with GEPA off, then the same with GEPA on. Each trial uses an independent sample of 50 HarmBench behaviors and logs the **jailbreak fraction** to `outputs/trial_summaries.txt` and stdout. On a remote GPU (e.g. `ssh ubuntu@216.81.248.162 -i private_key.pem`), copy the repo and run:
  ```bash
  python3 run_trials.py
  ```
  Or from your machine: `./run_trials_remote.sh private_key.pem /path/to/project/on/server`. If you see unsloth errors about `PreTrainedConfig` or `RopeParameters`, use compatible versions (e.g. `pip install --upgrade unsloth` or check unsloth/transformers compatibility).

  No API key is required for the judge. If you set **`defense.USE_GEPA_DEFENDER: true`** in `configs/default.yaml`, install [Ollama](https://ollama.com) and run `ollama pull llama3:8b` so the GEPA reflector can run locally.

---

## Running on Google Colab

**GPU trials (recommended):** Use a **T4 GPU** runtime (or higher). In Colab cells:

1. **Reset and clone** (use `%%bash` so the cell runs as shell, not Python):

   ```bash
   %%bash
   cd /content
   rm -rf prompt-optimization
   git clone -b shiv-new_project_repo https://github.com/05rentao/prompt-optimization.git
   cd prompt-optimization
   ```

2. **Install from requirements:**

   ```bash
   %%bash
   cd /content/prompt-optimization
   pip install -r requirements.txt
   ```

3. **Run GPU trials** (GEPA off, then GEPA on; in-loop eval every 10 steps; outputs in `outputs/`):

   ```bash
   %%bash
   cd /content/prompt-optimization
   python run_trials_gpu.py
   ```

No API key is needed; the judge is the local HarmBench classifier. For the notebook flow, open **colab_experiment.ipynb**, set T4 GPU, and run Setup → Environment → Imports → Execution.

---

## Interpreting results

- **run_experiment.py**: Outputs a single CSV under `outputs/experiment_*.csv` (epoch, behavior, adversary_attack, target_response, reward, is_jailbroken, explanation, system_prompt, use_gepa_defender). Use `system_prompt` and `use_gepa_defender` to compare success rates with/without GEPA.
- **run_trials.py**: Writes `outputs/trial_summaries.txt` with jailbreak fraction per trial (GEPA off vs on). Same seed pattern (42 off, 123 on) for reproducible behavior sampling.
- **run_trials_gpu.py**: Produces `outputs/eval_during_training_*.csv` (in-loop eval every 10 steps) and `outputs/adversary_eval_*.csv` (post-trial eval). Trial summaries also go to `outputs/trial_summaries.txt`.

---

## Config summary

All parameters live in **configs/default.yaml**; **src/builders.py** uses them to construct adversary and target (with sensible defaults if keys are missing).

| Section     | Key example   | Purpose |
|------------|----------------|---------|
| `adversary` | `model_name`, `learning_rate`, `lora_r`, `lora_alpha`, `max_seq_length` | Red-team LM and training |
| `evaluator` | `model_name`  | Local HarmBench classifier (default `cais/HarmBench-Llama-2-13b-cls`) |
| `defense`   | `USE_GEPA_DEFENDER`, `model_name`, `evolution_steps` | GEPA toggle; when true, BlueTeam (Ollama llama3:8b) evolves system prompt on jailbreak |
| `target`    | `model_name`  | Target model name (for when you use a real model) |
| `training`  | `epochs`, `batch_size`, `harmbench_limit` | Loop and data size |

---

## License / course

For STAT 4830 use. Adversarial and harmful behavior data are for research and course purposes only.
