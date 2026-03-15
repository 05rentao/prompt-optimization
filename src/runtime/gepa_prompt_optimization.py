"""Reusable GEPA prompt-optimization helpers for refusal-style objectives."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import gepa.optimize_anything as oa
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

from src.experiments.evaluators import compute_refusal_score

from .interfaces import GenerationRequest, GenerationSession
from .openai_reflection_gateway import OpenAIReflectionGateway


@dataclass
class GepaPromptOptimizationConfig:
    """Runtime-agnostic configuration for GEPA prompt optimization."""

    max_metric_calls: int
    max_tokens: int
    temperature: float
    reflection_model_name: str
    objective_text: str = (
        "Optimize the system prompt so the model consistently refuses harmful requests "
        "while remaining concise and policy-aligned."
    )


class GepaRefusalEvaluator:
    """Callable GEPA evaluator that scores candidate prompts by refusal score."""

    def __init__(
        self,
        target_session: GenerationSession,
        device: str,
        max_tokens: int,
        temperature: float,
        optimizer_trace: list[dict[str, Any]],
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.target_session = target_session
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.optimizer_trace = optimizer_trace
        self.logger = logger

    def __call__(self, candidate: dict[str, Any], example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """Score one candidate on one dataset example for GEPA."""
        system_prompt = candidate.get("system_prompt", "")
        user_prompt = example["input"]
        start = time.perf_counter()
        response = self.target_session.generate(
            GenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
            device=self.device,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        refusal_score = compute_refusal_score(response)
        score = float(refusal_score)
        side_info = {
            "response_preview": response[:300],
            "refusal_score": float(refusal_score),
            "latency_ms": float(elapsed_ms),
        }

        if self.logger is not None:
            try:
                self.logger(
                    f"input={user_prompt[:200]} | refusal_score={refusal_score:.2f} | "
                    f"latency_ms={elapsed_ms:.1f} | response={response[:200]}"
                )
            except Exception:
                pass

        self.optimizer_trace.append(
            {
                "timestamp": time.time(),
                "score": score,
                "latency_ms": elapsed_ms,
                "input_preview": user_prompt[:100],
            }
        )
        return score, side_info


def run_gepa_prompt_optimization(
    cfg: GepaPromptOptimizationConfig,
    target_session: GenerationSession,
    reflection_gateway: OpenAIReflectionGateway,
    device: str,
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
    baseline_system_prompt: str,
) -> tuple[Any, list[dict[str, Any]], float]:
    """Run GEPA optimization and return result object, trace, and runtime seconds."""
    gepa_train = [{"input": ex["prompt"], "id": ex["id"]} for ex in train_data]
    gepa_val = [{"input": ex["prompt"], "id": ex["id"]} for ex in val_data]
    optimizer_trace: list[dict[str, Any]] = []
    evaluator = GepaRefusalEvaluator(
        target_session=target_session,
        device=device,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        optimizer_trace=optimizer_trace,
        logger=oa.log,
    )

    gepa_config = GEPAConfig(
        engine=EngineConfig(max_metric_calls=cfg.max_metric_calls),
        reflection=ReflectionConfig(reflection_lm=f"openai/{cfg.reflection_model_name}"),
    )

    start_run = time.time()
    with reflection_gateway.bind_openai_env():
        result = optimize_anything(
            seed_candidate={"system_prompt": baseline_system_prompt},
            evaluator=evaluator,
            dataset=gepa_train,
            valset=gepa_val,
            objective=cfg.objective_text,
            config=gepa_config,
        )
    run_seconds = time.time() - start_run
    return result, optimizer_trace, run_seconds

