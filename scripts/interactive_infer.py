#!/usr/bin/env python3
"""One-shot or REPL chat inference: local Transformers or OpenAI-compatible HTTP (e.g. vLLM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.runtime import (
    GenerationRequest,
    GenerationSession,
    LocalHFConfig,
    OpenAITargetConfig,
    RuntimeCatalog,
    load_default_config,
    resolve_config_path,
    resolve_reflection_env_overrides,
)


def _read_text(path: Path) -> str:
    return path.expanduser().read_text(encoding="utf-8").strip()


def _resolve_default_model(backend: str, defaults: dict) -> str:
    models = defaults["runtime"]["models"]
    if backend == "http":
        return models["reflection_model_name"]
    return models["target_model_name"]


def _build_session(
    backend: str,
    model_id: str,
    max_new_tokens: int,
    use_4bit: bool,
    defaults: dict,
    base_url: str | None,
    api_key: str | None,
) -> GenerationSession:
    if backend == "local":
        return RuntimeCatalog.build_target_session(
            LocalHFConfig(model_id=model_id, use_4bit=use_4bit, max_new_tokens=max_new_tokens)
        )
    bu, key = resolve_reflection_env_overrides(defaults)
    if base_url is not None:
        bu = base_url
    if api_key is not None:
        key = api_key
    return RuntimeCatalog.build_openai_target_session(
        OpenAITargetConfig(base_url=bu, api_key=key, model_id=model_id)
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a single chat completion or an interactive REPL (local HF or HTTP target).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path (default: PROMPT_OPT_CONFIG_PATH or configs/default.yaml).",
    )
    p.add_argument(
        "--backend",
        choices=("local", "http"),
        default="local",
        help="local: Hugging Face weights on this machine; http: OpenAI-compatible server (vLLM).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model id: HF repo id for --backend local; served name for --backend http. "
        "Default: target_model_name (local) or reflection_model_name (http) from config.",
    )
    p.add_argument("--device", default=None, help="Device for local inference (default: cuda if available else cpu).")
    p.add_argument("--no-4bit", action="store_true", help="Load local model in full precision (more VRAM).")
    p.add_argument("--base-url", default=None, help="Override vLLM base URL for --backend http.")
    p.add_argument("--api-key", default=None, help="Override API key for --backend http.")
    p.add_argument(
        "--system",
        "-s",
        default=None,
        help="System message. Use @path to read from a file.",
    )
    p.add_argument(
        "--prompt",
        "-p",
        "--user",
        dest="user_prompt",
        default=None,
        help="User message. Use @path to read from a file.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="REPL: repeatedly enter system (optional) and user text until 'quit' or EOF.",
    )
    return p.parse_args()


def _resolve_prompt(raw: str | None, label: str) -> str:
    if raw is None:
        return ""
    s = raw.strip()
    if s.startswith("@"):
        return _read_text(Path(s[1:]))
    return s


def _infer_one(
    session: GenerationSession,
    device: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    req = GenerationRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return session.generate(req, device=device)


def _repl(
    session: GenerationSession,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    initial_system: str,
) -> None:
    system_prompt = initial_system
    print("Interactive mode. Commands: empty user line = skip turn; 'quit' or 'exit' to stop.", file=sys.stderr)
    while True:
        try:
            sys_line = input("system (optional, blank to keep): ").strip()
        except EOFError:
            print(file=sys.stderr)
            break
        if sys_line.lower() in ("quit", "exit"):
            break
        if sys_line:
            system_prompt = sys_line

        try:
            user_line = input("user: ").strip()
        except EOFError:
            print(file=sys.stderr)
            break
        if user_line.lower() in ("quit", "exit"):
            break
        if not user_line:
            continue

        out = _infer_one(session, device, system_prompt, user_line, max_new_tokens, temperature, top_p)
        print("\n--- assistant ---\n")
        print(out)
        print("\n---\n")


def main() -> None:
    args = _parse_args()
    config_path = resolve_config_path(args.config)
    defaults = load_default_config(config_path)
    model_id = args.model or _resolve_default_model(args.backend, defaults)
    if args.backend == "http":
        device = args.device or "cpu"
    else:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    session = _build_session(
        args.backend,
        model_id,
        args.max_new_tokens,
        use_4bit=not args.no_4bit,
        defaults=defaults,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.interactive:
        initial = _resolve_prompt(args.system, "system")
        _repl(session, device, args.max_new_tokens, args.temperature, args.top_p, initial)
        return

    system_prompt = _resolve_prompt(args.system, "system")
    user_prompt = _resolve_prompt(args.user_prompt, "user")
    if not user_prompt:
        print("error: provide --prompt / -p (or use --interactive).", file=sys.stderr)
        sys.exit(2)

    text = _infer_one(
        session,
        device,
        system_prompt,
        user_prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
