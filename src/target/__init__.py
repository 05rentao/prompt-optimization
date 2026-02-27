"""Target model used for red-team evaluation (the model being attacked)."""
from src.target.ollama_target import make_ollama_target, make_target_model, make_mock_target
from src.target.hf_target import make_hf_target

__all__ = ["make_ollama_target", "make_target_model", "make_mock_target", "make_hf_target"]
