from __future__ import annotations

import random as _random
from typing import Any

import pandas as pd
from datasets import load_dataset

from .types import GepaExampleRow, HarmBenchExampleRow

# 100 fixed indices into the raw (unshuffled) HarmBench standard/train split.
# Generated once at import time with seed 42; never changes between runs or scripts.
EVAL_SUBSET_INDICES: list[int] = sorted(_random.Random(42).sample(range(400), 100))


def set_eval_seed(seed: int = 42) -> None:
    """Reset all RNG sources immediately before an evaluation pass.

    Must be called inside the eval function (not once at startup) so that the
    seed is in effect for each individual eval call. Calling it at startup lets
    the seed drift across the run, making later evals non-reproducible.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fixed_eval_subset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    hf_token: str,
    indices: list[int] = EVAL_SUBSET_INDICES,
) -> list[HarmBenchExampleRow]:
    """Load the fixed seeded eval subset from the raw (unshuffled) dataset.

    No shuffle is applied so that each index always resolves to the same prompt
    regardless of training seed or run order. All scripts that call this function
    will evaluate on the same 100 HarmBench prompts, making results comparable.
    """
    ds = load_dataset(dataset_name, name=dataset_config, split=split, token=hf_token)
    valid = [i for i in indices if i < len(ds)]
    return [normalize_harmbench_record(dict(ds[i]), i) for i in valid]

PROMPT_KEYS = [
    "prompt",
    "user_prompt",
    "instruction",
    "query",
    "goal",
    "behavior",
    "input",
]
ID_KEYS = ["id", "sample_id", "uid", "idx"]


def pick_first_available(dct: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in dct and dct[key] is not None:
            return dct[key]
    return default


def normalize_harmbench_record(record: dict[str, Any], idx: int) -> HarmBenchExampleRow:
    prompt = pick_first_available(record, PROMPT_KEYS)
    if prompt is None:
        raise ValueError(f"Could not find prompt text in record keys: {list(record.keys())}")

    rec_id = pick_first_available(record, ID_KEYS, default=idx)
    return {
        "id": str(rec_id),
        "prompt": str(prompt).strip(),
        "is_harmful_request": True,
    }


def load_harmbench_subset(
    dataset_name: str,
    dataset_config: str | None, 
    split: str,
    train_size: int,
    val_size: int,
    seed: int,
    hf_token: str,
) -> tuple[list[HarmBenchExampleRow], list[HarmBenchExampleRow], pd.DataFrame]:
    
    ds = load_dataset(dataset_name, name=dataset_config, split=split, token=hf_token)
    # dataset_dict = load_dataset(walledai/HarmBench, name=standard, split=train, token=hf_token)
    # train_df = dataset_dict['train'].to_pandas()


    total_needed = train_size + val_size
    if len(ds) < total_needed:
        raise ValueError(f"Need {total_needed} samples, but only found {len(ds)}.")

    # Shuffle and normalize
    shuffled = ds.shuffle(seed=seed).select(range(total_needed))
    normalized = [normalize_harmbench_record(shuffled[i], i) for i in range(total_needed)]

    return normalized[:train_size], normalized[train_size:], pd.DataFrame(normalized)
    # returns train_data, val_data, preview_df


def harmbench_to_gepa_examples(examples: list[dict[str, Any]]) -> list[GepaExampleRow]:
    """Map normalized HarmBench examples to GEPA optimizer example schema."""
    return [{"id": str(ex["id"]), "input": str(ex["prompt"])} for ex in examples]


def build_gepa_prompt_dataset(
    prompts: list[str],
    train_ratio: float = 0.7,
    min_val_examples: int = 1,
) -> tuple[list[GepaExampleRow], list[GepaExampleRow]]:
    """Build GEPA train/val datasets from prompt text list."""
    unique_prompts = list(dict.fromkeys(prompts))
    if not unique_prompts:
        return [], []
    split_idx = max(1, int(len(unique_prompts) * train_ratio))
    train = [{"id": f"train_{i}", "input": text} for i, text in enumerate(unique_prompts[:split_idx])]
    val = [{"id": f"val_{i}", "input": text} for i, text in enumerate(unique_prompts[split_idx:])]
    if not val:
        val = train[: min(max(min_val_examples, 1), len(train))]
    return train, val
