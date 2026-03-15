from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import load_dataset

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


def normalize_harmbench_record(record: dict[str, Any], idx: int) -> dict[str, Any]:
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    
    ds = load_dataset(dataset_name, name=dataset_config, split=split, token=hf_token)

    total_needed = train_size + val_size
    if len(ds) < total_needed:
        raise ValueError(f"Need {total_needed} samples, but only found {len(ds)}.")

    # Shuffle and normalize
    shuffled = ds.shuffle(seed=seed).select(range(total_needed))
    normalized = [normalize_harmbench_record(shuffled[i], i) for i in range(total_needed)]

    return normalized[:train_size], normalized[train_size:], pd.DataFrame(normalized)
    # returns train_data, val_data, preview_df
