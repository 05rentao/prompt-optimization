"""Target model used for red-team evaluation (the model being attacked)."""
from src.target.ollama_target import make_ollama_target, make_target_model

__all__ = ["make_ollama_target", "make_target_model"]
