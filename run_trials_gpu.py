"""
GPU trial: one run with GEPA off (defender unchanged), batch_size=20, 150 HarmBench prompts.
Each REINFORCE step randomly samples 20 of the 150 behaviors, runs rollouts, then one SGD update.

After training, an independent eval runs: 50 random HarmBench prompts (fixed seed), adversary
generates jailbreak prompts, target responds, HarmBench judge scores each; jailbreak fraction
and per-prompt results are written to CSV (adversary_eval_summary.csv + adversary_eval_detail_*.csv).

Usage on GPU server:
  cd /path/to/project && python run_trials_gpu.py
"""
import copy
import csv
import os
from datetime import datetime
import yaml
import torch
from src.adversary.policy import RedTeamPolicy
from src.data.harmbench_loader import HarmBenchLoader
from src.evaluator.judge import get_judge
from src.target.ollama_target import make_target_model
from src.training.loop import TrainingLoop, BASE_SYSTEM_PROMPT
from src.training.logger import log_trial_summary

HARMBENCH_TOTAL = 150
BATCH_SIZE = 20
MAX_STEPS = 150   # number of batch updates (each uses 20 random samples)
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
    return make_target_model(
        use_ollama=t.get("use_ollama", False),
        ollama_model=t.get("ollama_model", "llama3:8b"),
        ollama_url=t.get("ollama_url", "http://localhost:11434"),
    )


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


def main():
    config = load_config()
    cfg = copy.deepcopy(config)
    cfg["defense"]["USE_GEPA_DEFENDER"] = False  # defender does not change
    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["harmbench_limit"] = HARMBENCH_TOTAL

    print("📊 Loading 150 HarmBench behaviors...")
    behaviors = HarmBenchLoader.load(limit=HARMBENCH_TOTAL, seed=42)

    print(f"\n{'='*60}")
    print(f"  GPU trial: GEPA OFF, batch_size={BATCH_SIZE}, {len(behaviors)} behaviors")
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
    print(f"\n📊 GPU trial (GEPA off) -> jailbreak_fraction={frac:.4f} (steps={steps}, rollouts={rollouts}, elapsed={elapsed:.1f}s)")

    path = log_trial_summary("gpu_trial_gepa_off", frac, steps, elapsed)
    print(f"   Logged to {path}")

    # Independent eval: 50 random HarmBench prompts, adversary -> target -> judge, CSV output
    print(f"\n📋 Running adversary eval ({EVAL_NUM_PROMPTS} random HarmBench prompts, seed={EVAL_SEED})...")
    judge = get_judge(cfg.get("evaluator", {}).get("model_name", "cais/HarmBench-Llama-2-13b-cls"))
    target = get_target_from_config(cfg)
    eval_frac, summary_csv, detail_csv = run_adversary_eval(
        adversary, target, judge, cfg, trial_name="gpu_trial_gepa_off", num_prompts=EVAL_NUM_PROMPTS, seed=EVAL_SEED
    )
    print(f"   Eval jailbreak fraction: {eval_frac:.4f} -> summary: {summary_csv}, detail: {detail_csv}")

    print("\n✅ Done. Check outputs/trial_summaries.txt and outputs/adversary_eval_*.csv.")


if __name__ == "__main__":
    main()
