"""Runtime construction catalog used by experiment entrypoints."""

from __future__ import annotations

import os
from dataclasses import replace

from .config import HarmbenchJudgeConfig, LocalHFConfig, OpenAIReflectionConfig, OpenAITargetConfig, UnslothAdversaryConfig
from .harmbench_judge_runtime import HarmbenchJudgeRuntime
from .interfaces import GenerationSession
from .local_hf_runtime import LocalHFChatRuntime
from .openai_reflection_gateway import OpenAIReflectionGateway
from .openai_target_runtime import OpenAIChatTargetRuntime
from .unsloth_adversary_runtime import UnslothAdversaryRuntime


class RuntimeCatalog:
    """Builds typed runtime sessions and reflection gateways."""

    @staticmethod
    def build_target_session(cfg: LocalHFConfig) -> GenerationSession:
        """Create a target-model generation session."""
        return GenerationSession(runtime=LocalHFChatRuntime(cfg))

    @staticmethod
    def build_openai_target_session(cfg: OpenAITargetConfig) -> GenerationSession:
        """Create a target session backed by an OpenAI-compatible HTTP endpoint (e.g. vLLM)."""
        return GenerationSession(runtime=OpenAIChatTargetRuntime(cfg))

    @staticmethod
    def build_adversary_session(cfg: UnslothAdversaryConfig) -> GenerationSession:
        """Create an adversary-model generation session."""
        return GenerationSession(runtime=UnslothAdversaryRuntime(cfg))

    @staticmethod
    def build_judge_session(cfg: HarmbenchJudgeConfig) -> GenerationSession:
        """Create a HarmBench judge session."""
        env = os.environ.get("JUDGE_LOAD_IN_4BIT", "").lower()
        if env in ("0", "false", "no"):
            cfg = replace(cfg, load_in_4bit=False)
        return GenerationSession(runtime=HarmbenchJudgeRuntime(cfg))

    @staticmethod
    def build_reflection_gateway(cfg: OpenAIReflectionConfig) -> OpenAIReflectionGateway:
        """Create a frozen OpenAI-compatible reflection gateway."""
        return OpenAIReflectionGateway(cfg)

