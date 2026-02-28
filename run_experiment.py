"""
Single training run: load config, build adversary + target + data, run TrainingLoop.
No step/time cap; batch_size and epochs from config.
"""
from src.utils.config import load_config
from src.builders import build_adversary, get_target_from_config
from src.data.harmbench_loader import HarmBenchLoader
from src.training.loop import TrainingLoop


def main():
    config = load_config()

    print("🚀 Loading Red Team (Adversary)...")
    adversary = build_adversary(config)

    print("📊 Loading HarmBench behaviors...")
    behaviors = HarmBenchLoader.load(limit=config["training"]["harmbench_limit"])

    target_model = get_target_from_config(config)

    trainer = TrainingLoop(
        adversary=adversary,
        target_model=target_model,
        behaviors=behaviors,
        config=config,
    )
    trainer.run()

if __name__ == "__main__":
    main()