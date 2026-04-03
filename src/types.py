from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict
from typing_extensions import NotRequired


@dataclass(frozen=True)
class HarmBenchDatasetConfig:
    dataset_name: str = "walledai/HarmBench"
    dataset_config: str | None = "standard"
    split: str = "train"
    train_size: int = 100
    val_size: int = 100
    seed: int = 42
    hf_token: str = ""


@dataclass
class RunManifest:
    mode: str
    runtime_profile: str
    seed: int
    dataset: dict[str, Any]
    models: dict[str, Any]
    budget: dict[str, Any]
    endpoints: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    #: CLI args, config file path, reflection URL, and env flags (see ``build_config_snapshot``).
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HarmBenchExampleRow(TypedDict):
    """Normalized HarmBench example schema shared across runs."""

    id: str
    prompt: str
    is_harmful_request: bool


class XSTestExampleRow(TypedDict):
    """Normalized XSTest example schema for overrefusal benchmarking."""

    id: str
    prompt: str
    is_safe: bool
    category: str


class GepaExampleRow(TypedDict):
    """GEPA example schema expected by optimize_anything evaluators."""

    id: str
    input: str


class CoevEvalOutputRow(TypedDict):
    """Row-level evaluation output schema for CoEV-style pipelines."""

    id: str
    prompt: str
    adversary_prompt: str
    target_response: str
    refusal_score: float
    asr_score: float
    adversary_latency_ms: float
    target_latency_ms: float
    latency_ms_total: float


class CoevTrainingLogRow(TypedDict):
    """Per-iteration training log row schema for CoEV pipelines."""

    stage: int
    iter: int
    dataset_index: int
    attacker_instruction: str
    defense_prompt: str
    orig_prompt: str
    adv_prompt: str
    target_resp: str
    reward: float
    loss: float
    verdict: str
    # Present when adversary REINFORCE uses batch_size > 1 (mean reward in `reward`).
    max_reward: NotRequired[float]
    batch_size: NotRequired[int]
    adversary_policy: NotRequired[str]
    target_queries: NotRequired[int]
    rejection_sampling: NotRequired[bool]


class CoevStageMetricRow(TypedDict, total=False):
    """Per-stage metrics row schema for staged CoEV training."""

    stage: int
    phase: str
    score: float
    refusal_rate: float
    asr: float
    mean_refusal_score: float
    aggregate_score: float
    latency_ms_mean: float


class OptimizerTraceRow(TypedDict, total=False):
    """Optimizer trace row schema for GEPA optimization trajectories."""

    timestamp: float
    stage: int
    role: str
    score: float
    input_preview: str
    latency_ms: float
