"""Public runtime API surface for experiment orchestration."""

from .catalog import RuntimeCatalog
from .config import (
    CoevConfig,
    GepaOptimizationConfig,
    HarmbenchJudgeConfig,
    LocalHFConfig,
    ModelConfig,
    OpenAIReflectionConfig,
    OpenAITargetConfig,
    TargetModelConfig,
    UnslothAdversaryConfig,
)
from .defaults import load_default_config
from .target_factory import (
    build_local_hf_target_session,
    build_reflection_gateway_for_defaults,
    build_vllm_target_session,
    resolve_reflection_env_overrides,
)
from .env import resolve_hf_token, scoped_env
from .evaluation import EvaluationBatchResult, EvaluationConfig, EvaluatedSample, EvaluationResult, evaluate_examples, evaluate_outputs
from .gepa_prompt_optimization import (
    DualRoleGepaContext,
    DualRoleGepaOptimizationResult,
    DualRoleGepaPromptOptimizationConfig,
    GepaPromptOptimizationConfig,
    run_dual_role_gepa_prompt_optimization,
    run_gepa_prompt_optimization,
)
from .interfaces import GenerationRequest, GenerationSession, JudgeRuntime, LoRABridge, ReflectionGateway, TargetRuntime

__all__ = [
    "GenerationRequest",
    "GenerationSession",
    "HarmbenchJudgeConfig",
    "JudgeRuntime",
    "LocalHFConfig",
    "ModelConfig",
    "TargetModelConfig",
    "CoevConfig",
    "GepaOptimizationConfig",
    "LoRABridge",
    "OpenAIReflectionConfig",
    "OpenAITargetConfig",
    "ReflectionGateway",
    "RuntimeCatalog",
    "TargetRuntime",
    "UnslothAdversaryConfig",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluatedSample",
    "EvaluationBatchResult",
    "evaluate_examples",
    "evaluate_outputs",
    "GepaPromptOptimizationConfig",
    "DualRoleGepaPromptOptimizationConfig",
    "DualRoleGepaContext",
    "DualRoleGepaOptimizationResult",
    "run_gepa_prompt_optimization",
    "run_dual_role_gepa_prompt_optimization",
    "load_default_config",
    "build_vllm_target_session",
    "build_local_hf_target_session",
    "build_reflection_gateway_for_defaults",
    "resolve_reflection_env_overrides",
    "resolve_hf_token",
    "scoped_env",
]

