"""
Build adversary and target from config. Single place for config key names and defaults.
Keeps entry points (run_experiment, run_trials, run_trials_gpu) thin.
"""
from typing import Any, Callable, Dict

from src.adversary.policy import RedTeamPolicy
from src.target.hf_target import make_hf_target
from src.target.ollama_target import make_mock_target, make_target_model


def build_adversary(config: Dict[str, Any]) -> RedTeamPolicy:
    """
    Build RedTeamPolicy (adversary) from config['adversary'].

    Args:
        config: Full config dict; uses adversary.model_name, learning_rate, load_in_4bit.

    Returns:
        RedTeamPolicy instance.
    """
    adv = config.get("adversary", {})
    return RedTeamPolicy(
        model_name=adv.get("model_name", "unsloth/Qwen2.5-1.5B-Instruct"),
        lr=float(adv.get("learning_rate", 1e-5)),
        load_in_4bit=adv.get("load_in_4bit", False),
    )


def get_target_from_config(config: Dict[str, Any]) -> Callable[[str], str]:
    """
    Build target model callable from config['target'].

    - use_hf: True -> HF Qwen (hf_model_name, hf_load_4bit)
    - use_ollama: True -> Ollama (ollama_model, ollama_url)
    - else: mock target (echo-style)

    Args:
        config: Full config dict.

    Returns:
        Callable that takes a single string (user prompt) and returns the model response string.
    """
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
