from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from .types import GepaExampleRow, HarmBenchExampleRow

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
    """Load a shuffled prefix of HarmBench; split into train and optional val.

    ``val_size`` may be ``0`` to load only ``train_size`` rows (train list full, val empty).
    """
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


def harmbench_csv_row_to_prompt(row: pd.Series) -> str:
    """Build the full behavior text from a HarmBench behaviors CSV row.

    Contextual rows combine ``Behavior`` with ``ContextString`` the way benchmark
    suites typically present the full harmful request to models and judges.
    """
    behavior = str(row["Behavior"]).strip()
    raw_ctx = row.get("ContextString", "")
    if raw_ctx is None or (isinstance(raw_ctx, float) and pd.isna(raw_ctx)):
        return behavior
    ctx = str(raw_ctx).strip()
    if not ctx:
        return behavior
    return f"{behavior}\n\n{ctx}"


_REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_harmbench_csv_path(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve a CSV path relative to the repository root when not absolute."""
    root = repo_root or _REPO_ROOT
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def load_harmbench_behaviors_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a HarmBench behaviors CSV from a local path (relative paths use repo root)."""
    resolved = resolve_harmbench_csv_path(csv_path)
    return pd.read_csv(resolved)


def load_harmbench_csv_val_test_splits(
    csv_path: str | Path,
    *,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split behaviors into fixed validation and held-out test lists for adversary eval.

    Rows are shuffled deterministically with ``seed``, then the first ``val_size`` rows
    are validation (e.g. periodic ASR during training) and the next ``test_size`` rows
    are the held-out test set (e.g. baseline/final reporting).

    Each returned item is ``{"id": str, "prompt": str}`` compatible with
    ``adversary_run.evaluate_asr`` example dicts.
    """
    need = val_size + test_size
    df = load_harmbench_behaviors_csv(csv_path)
    if len(df) < need:
        raise ValueError(
            f"HarmBench CSV needs at least {need} rows for val_size={val_size} + test_size={test_size}, "
            f"got {len(df)}."
        )
    if "Behavior" not in df.columns:
        raise ValueError(f"CSV missing Behavior column; columns={list(df.columns)}")
    if "BehaviorID" not in df.columns:
        raise ValueError(f"CSV missing BehaviorID column; columns={list(df.columns)}")

    order = list(range(len(df)))
    rng = random.Random(seed)
    rng.shuffle(order)
    chosen = order[:need]

    def row_to_example(idx: int) -> dict[str, Any]:
        row = df.iloc[idx]
        pid = str(row["BehaviorID"]).strip()
        prompt = harmbench_csv_row_to_prompt(row)
        return {"id": pid, "prompt": prompt}

    val_rows = [row_to_example(chosen[i]) for i in range(val_size)]
    test_rows = [row_to_example(chosen[i]) for i in range(val_size, need)]
    return val_rows, test_rows
