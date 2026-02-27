"""
GPU trials: two runs to compare — (1) GEPA off (defender unchanged), (2) GEPA on.
Each trial: batch_size=4, 150 HarmBench prompts; each step samples 4 at random, rollouts, one SGD update.
After each trial, an independent eval runs: 50 random HarmBench prompts (fixed seed), adversary -> target -> judge;
jailbreak fraction and per-prompt results are written to CSV (adversary_eval_summary.csv + adversary_eval_detail_*.csv).

Usage on GPU server:
  cd /path/to/project && python run_trials_gpu.py
"""
import copy
import csv
import gc
import os
from datetime import datetime
import yaml
import torch
from src.adversary.policy import RedTeamPolicy
from src.data.harmbench_loader import HarmBenchLoader
from src.evaluator.judge import get_judge
from src.target.ollama_target import make_target_model, make_mock_target
from src.target.hf_target import make_hf_target
from src.training.loop import TrainingLoop, BASE_SYSTEM_PROMPT
from src.training.logger import log_trial_summary

HARMBENCH_TOTAL = 150
BATCH_SIZE = 4
MAX_STEPS = 100   # number of batch updates (each uses BATCH_SIZE random samples)
MAX_SECONDS = None  # no time cap; use MAX_STEPS

# Independent eval: 50 random HarmBench prompts, fixed seed (separate from training data)
EVAL_NUM_PROMPTS = 50
EVAL_SEED = 999
EVAL_SUMMARY_CSV = "outputs/adversary_eval_summary.csv"


def load_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def get_target_from_config(config):
    t = config.get("target", {})
    if t.get("use_hf", False):
        return make_hf_target(
            model_name=t.get("hf_model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
            load_in_4bit=t.get("hf_load_4bit", True),
        )
    if t.get("use_ollama", False):
        return make_target_model(
            use_ollama=True,
            ollama_model=t.get("ollama_model", "llama3:8b"),
            ollama_url=t.get("ollama_url", "http://localhost:11434"),
        )
    return make_mock_target()


def run_adversary_eval(adversary, target, judge, config, trial_name: str, num_prompts: int = EVAL_NUM_PROMPTS, seed: int = EVAL_SEED, output_dir: str = "outputs"):
    """
    Independent of training: load num_prompts random HarmBench behaviors (fixed seed), have the
    adversary generate a jailbreak prompt for each, run target -> judge, record results.
    Appends one row to adversary_eval_summary.csv and writes adversary_eval_detail_<timestamp>.csv.
    Returns jailbreak_fraction.
    """
    os.makedirs(output_dir, exist_ok=True)
    eval_behaviors = HarmBenchLoader.load(limit=num_prompts, seed=seed)
    rows = []
    for entry in eval_behaviors:
        behavior = entry["behavior"]
        with torch.no_grad():
            attack_prompt, _ = adversary.generate_attack(behavior)
        full_input = f"System: {BASE_SYSTEM_PROMPT}\nUser: {attack_prompt}"
        response = target(full_input)
        with torch.no_grad():
            eval_res = judge(behavior=behavior, response=response)
        is_jailbroken = getattr(eval_res, "is_jailbroken", "No")
        rows.append({
            "behavior": behavior,
            "adversary_attack": attack_prompt,
            "target_response": response,
            "is_jailbroken": is_jailbroken,
        })
    num_jailbroken = sum(1 for r in rows if (r["is_jailbroken"] or "").strip().lower() == "yes")
    jailbreak_fraction = num_jailbroken / len(rows) if rows else 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Append summary
    summary_path = os.path.join(output_dir, "adversary_eval_summary.csv")
    summary_exists = os.path.isfile(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "trial_name", "jailbreak_fraction", "num_jailbroken", "num_total"])
        if not summary_exists:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now().isoformat(),
            "trial_name": trial_name,
            "jailbreak_fraction": round(jailbreak_fraction, 4),
            "num_jailbroken": num_jailbroken,
            "num_total": len(rows),
        })
    # Detail CSV (one row per prompt)
    detail_path = os.path.join(output_dir, f"adversary_eval_detail_{trial_name}_{timestamp}.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["behavior", "adversary_attack", "target_response", "is_jailbroken"])
        w.writeheader()
        w.writerows(rows)
    return jailbreak_fraction, summary_path, detail_path


def run_one_trial(config, behaviors, use_gepa: bool, trial_name: str):
    """Run one GPU trial (GEPA on or off), then independent adversary eval; return summary and eval fraction."""
    cfg = copy.deepcopy(config)
    cfg["defense"]["USE_GEPA_DEFENDER"] = use_gepa
    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["harmbench_limit"] = len(behaviors)

    print(f"\n{'='*60}")
    print(f"  {trial_name}: GEPA={'ON' if use_gepa else 'OFF'}, batch_size={BATCH_SIZE}, {len(behaviors)} behaviors")
    print(f"  Each step: random sample of {BATCH_SIZE} -> rollouts -> one SGD update")
    print(f"  Max steps: {MAX_STEPS}")
    print("="*60)

    adversary = RedTeamPolicy(
        model_name=cfg["adversary"]["model_name"],
        lr=float(cfg["adversary"]["learning_rate"]),
        load_in_4bit=cfg["adversary"].get("load_in_4bit", False),
    )
    trainer = TrainingLoop(
        adversary=adversary,
        target_model=get_target_from_config(cfg),
        behaviors=behaviors,
        config=cfg,
    )
    summary = trainer.run(max_steps=MAX_STEPS, max_seconds=MAX_SECONDS, batch_size=BATCH_SIZE)

    frac = summary["jailbreak_fraction"]
    steps = summary["steps"]
    rollouts = summary.get("rollouts", steps)
    elapsed = summary["elapsed_seconds"]
    print(f"\n📊 {trial_name} -> jailbreak_fraction={frac:.4f} (steps={steps}, rollouts={rollouts}, elapsed={elapsed:.1f}s)")

    log_trial_summary(trial_name, frac, steps, elapsed)

    # Independent eval: 50 random HarmBench prompts, adversary -> target -> judge, CSV output
    print(f"\n📋 Running adversary eval for {trial_name} ({EVAL_NUM_PROMPTS} prompts, seed={EVAL_SEED})...")
    judge = get_judge(cfg.get("evaluator", {}).get("model_name", "cais/HarmBench-Llama-2-13b-cls"))
    target = get_target_from_config(cfg)
    eval_frac, summary_csv, detail_csv = run_adversary_eval(
        adversary, target, judge, cfg, trial_name=trial_name, num_prompts=EVAL_NUM_PROMPTS, seed=EVAL_SEED
    )
    print(f"   Eval jailbreak fraction: {eval_frac:.4f} -> detail: {detail_csv}")
    return {"train_frac": frac, "eval_frac": eval_frac, "steps": steps, "elapsed": elapsed}


def main():
    config = load_config()
    config["training"]["batch_size"] = BATCH_SIZE
    config["training"]["harmbench_limit"] = HARMBENCH_TOTAL

    print("📊 Loading two independent 150-behavior samples from HarmBench...")
    behaviors_off = HarmBenchLoader.load(limit=HARMBENCH_TOTAL, seed=42)
    behaviors_on = HarmBenchLoader.load(limit=HARMBENCH_TOTAL, seed=123)

    result_off = run_one_trial(config, behaviors_off, use_gepa=False, trial_name="gpu_trial_gepa_off")
    # Free GPU memory before loading second trial (avoids OOM when adversary + judge both on GPU)
    print("\n🧹 Freeing GPU memory before second trial...")
    free_gpu_memory()
    result_on = run_one_trial(config, behaviors_on, use_gepa=True, trial_name="gpu_trial_gepa_on")

    print("\n" + "="*60)
    print("  Comparison (GEPA off vs on)")
    print("="*60)
    print(f"  GEPA OFF: train jailbreak_frac={result_off['train_frac']:.4f}, eval jailbreak_frac={result_off['eval_frac']:.4f}")
    print(f"  GEPA ON:  train jailbreak_frac={result_on['train_frac']:.4f}, eval jailbreak_frac={result_on['eval_frac']:.4f}")
    print("\n✅ Done. Check outputs/trial_summaries.txt and outputs/adversary_eval_*.csv.")


if __name__ == "__main__":
    main()
