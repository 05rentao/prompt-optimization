"""Runtime construction catalog used by experiment entrypoints."""

from __future__ import annotations

from .config import HarmbenchJudgeConfig, LocalHFConfig, OpenAIReflectionConfig, UnslothAdversaryConfig
from .harmbench_judge_runtime import HarmbenchJudgeRuntime
from .interfaces import GenerationSession
from .local_hf_runtime import LocalHFChatRuntime
from .openai_reflection_gateway import OpenAIReflectionGateway
from .unsloth_adversary_runtime import UnslothAdversaryRuntime


class RuntimeCatalog:
    """Builds typed runtime sessions and reflection gateways."""

    @staticmethod
    def build_target_session(cfg: LocalHFConfig) -> GenerationSession:
        """Create a target-model generation session."""
        return GenerationSession(runtime=LocalHFChatRuntime(cfg))

    @staticmethod
    def build_adversary_session(cfg: UnslothAdversaryConfig) -> GenerationSession:
        """Create an adversary-model generation session."""
        return GenerationSession(runtime=UnslothAdversaryRuntime(cfg))

    @staticmethod
    def build_judge_session(cfg: HarmbenchJudgeConfig) -> GenerationSession:
        """Create a HarmBench judge session."""
        return GenerationSession(runtime=HarmbenchJudgeRuntime(cfg.model_id))

    @staticmethod
    def build_reflection_gateway(cfg: OpenAIReflectionConfig) -> OpenAIReflectionGateway:
        """Create a frozen OpenAI-compatible reflection gateway."""
        return OpenAIReflectionGateway(cfg)

