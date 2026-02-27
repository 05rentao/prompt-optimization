import yaml
from src.adversary.policy import RedTeamPolicy
from src.data.harmbench_loader import HarmBenchLoader
from src.target.ollama_target import make_target_model
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
        lr=float(config['adversary']['learning_rate']),
        load_in_4bit=config['adversary'].get('load_in_4bit', False),
    )
    
    # 2. Load Data
    print("📊 Loading HarmBench behaviors...")
    behaviors = HarmBenchLoader.load(limit=config['training']['harmbench_limit'])

    # 3. Target Model: Ollama Llama (e.g. llama3:8b) if use_ollama else mock
    t = config.get("target", {})
    target_model = make_target_model(
        use_ollama=t.get("use_ollama", False),
        ollama_model=t.get("ollama_model", "llama3:8b"),
        ollama_url=t.get("ollama_url", "http://localhost:11434"),
    )

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