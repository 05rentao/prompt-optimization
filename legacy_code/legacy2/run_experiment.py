import yaml
from src.adversary.policy import RedTeamPolicy
from src.data.harmbench_loader import HarmBenchLoader
from src.training.loop import TrainingLoop

def load_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # Judge is local cais/HarmBench-Llama-2-13b-cls (evaluator.model_name); no API key needed

    # 1. Load Red Team with config params
    print("🚀 Loading Red Team (Adversary)...")
    adversary = RedTeamPolicy(
        model_name=config['adversary']['model_name'],
        lr=float(config['adversary']['learning_rate'])
    )
    
    # 2. Load Data
    print("📊 Loading HarmBench behaviors...")
    behaviors = HarmBenchLoader.load(limit=config['training']['harmbench_limit'])

    # 3. Target Model (Keep as mock for testing, then swap to real Llama)
    def target_model(text): 
        # For testing, occasionally return a "bad" response so the reward isn't always 0
        if "jailbreak" in text.lower():
            return "Sure, here is how you do something harmful..."
        return "I cannot fulfill this request."

    # 4. Start Loop - passing the config dict is crucial for the internal logic
    trainer = TrainingLoop(
        adversary=adversary, 
        target_model=target_model, 
        behaviors=behaviors,
        config=config
    )
    
    trainer.run()

if __name__ == "__main__":
    main()