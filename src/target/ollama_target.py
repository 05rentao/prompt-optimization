"""
Target model callable: use a simple Llama via Ollama (e.g. llama3:8b) so the same
local setup works for both the target and the GEPA defender.
"""
import urllib.request
import urllib.error
import json

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3:8b"
TIMEOUT_SEC = 120


def make_ollama_target(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_OLLAMA_URL, timeout: int = TIMEOUT_SEC):
    """
    Return a callable that sends full_input to Ollama and returns the model reply.
    full_input is typically "System: ...\\nUser: ..." (same format as the training loop).
    """

    def target_model(full_input: str) -> str:
        try:
            data = json.dumps({
                "model": model,
                "prompt": full_input,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url.rstrip('/')}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            return (out.get("response") or "").strip()
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            return f"[Ollama error: {e}. Is Ollama running with model {model}?]"

    return target_model


def make_mock_target():
    """Mock target for testing: jailbreak keyword -> harmful-looking reply, else refusal."""

    def target_model(text: str) -> str:
        if "jailbreak" in text.lower():
            return "Sure, here is how you do something harmful..."
        return "I cannot fulfill this request."

    return target_model


def make_target_model(use_ollama: bool = False, ollama_model: str = DEFAULT_MODEL, ollama_url: str = DEFAULT_OLLAMA_URL):
    """
    Return the target callable from config-like options.
    use_ollama=True -> real Llama via Ollama; otherwise mock.
    """
    if use_ollama:
        return make_ollama_target(model=ollama_model, base_url=ollama_url)
    return make_mock_target()
