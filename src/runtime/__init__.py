"""Public runtime API surface for experiment orchestration."""

from .catalog import RuntimeCatalog
from .config import (
    CoevConfig,
    GepaOptimizationConfig,
    HarmbenchJudgeConfig,
    LocalHFConfig,
    ModelConfig,
    OpenAIReflectionConfig,
    TargetModelConfig,
    UnslothAdversaryConfig,
)
from .defaults import load_default_config
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
    "resolve_hf_token",
    "scoped_env",
]

