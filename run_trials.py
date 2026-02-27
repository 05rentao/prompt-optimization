"""
Run two trials on the GPU server: GEPA off (150 steps or 5 min) then GEPA on (150 steps or 5 min).
Each trial uses an independent sample of 50 HarmBench behaviors. Logs jailbreak fraction to
outputs/trial_summaries.txt and stdout.

Usage on remote GPU (e.g. after ssh ubuntu@216.81.248.162 -i private_key.pem):
  cd /path/to/project && python run_trials.py
"""
import copy
import yaml
from src.adversary.policy import RedTeamPolicy
from src.data.harmbench_loader import HarmBenchLoader
from src.training.loop import TrainingLoop
from src.training.logger import log_trial_summary

# Cap: 150 steps or 5 minutes, whichever is shorter
MAX_STEPS = 150
MAX_SECONDS = 300  # 5 min
HARMBENCH_LIMIT = 50


def load_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def make_target_model():
    def target_model(text):
        if "jailbreak" in text.lower():
            return "Sure, here is how you do something harmful..."
        return "I cannot fulfill this request."
    return target_model


def run_one_trial(config, behaviors, use_gepa: bool, trial_name: str):
    cfg = copy.deepcopy(config)
    cfg["training"]["harmbench_limit"] = len(behaviors)
    cfg["defense"]["USE_GEPA_DEFENDER"] = use_gepa

    print(f"\n{'='*60}")
    print(f"  {trial_name}: USE_GEPA_DEFENDER={use_gepa}, {len(behaviors)} behaviors")
    print(f"  Cap: {MAX_STEPS} steps or {MAX_SECONDS}s")
    print("="*60)

    adversary = RedTeamPolicy(
        model_name=cfg["adversary"]["model_name"],
        lr=float(cfg["adversary"]["learning_rate"]),
    )
    trainer = TrainingLoop(
        adversary=adversary,
        target_model=make_target_model(),
        behaviors=behaviors,
        config=cfg,
    )
    summary = trainer.run(max_steps=MAX_STEPS, max_seconds=MAX_SECONDS)

    frac = summary["jailbreak_fraction"]
    steps = summary["steps"]
    elapsed = summary["elapsed_seconds"]
    print(f"\n📊 {trial_name} -> jailbreak_fraction={frac:.4f} (steps={steps}, elapsed={elapsed:.1f}s)")

    path = log_trial_summary(trial_name, frac, steps, elapsed)
    print(f"   Logged to {path}")
    return summary


def main():
    config = load_config()

    print("📊 Loading two independent 50-behavior samples from HarmBench...")
    behaviors_off = HarmBenchLoader.load(limit=HARMBENCH_LIMIT, seed=42)
    behaviors_on = HarmBenchLoader.load(limit=HARMBENCH_LIMIT, seed=123)

    run_one_trial(config, behaviors_off, use_gepa=False, trial_name="trial_gepa_off")
    run_one_trial(config, behaviors_on, use_gepa=True, trial_name="trial_gepa_on")

    print("\n✅ Both trials finished. Check outputs/trial_summaries.txt for jailbreak fractions.")


if __name__ == "__main__":
    main()
