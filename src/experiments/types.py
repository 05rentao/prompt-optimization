from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
