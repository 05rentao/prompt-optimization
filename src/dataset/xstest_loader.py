"""Load and normalize XSTest dataset for overrefusal benchmarking."""

from __future__ import annotations

from typing import Any, TypedDict

from datasets import load_dataset


class XSTestExampleRow(TypedDict):
    """Normalized XSTest example schema."""

    id: str
    prompt: str
    is_safe: bool
    category: str


def normalize_xstest_record(record: dict[str, Any], idx: int) -> XSTestExampleRow:
    """Convert raw XSTest record to normalized schema."""
    return {
        "id": str(record.get("id", idx)),
        "prompt": str(record.get("prompt") or record.get("text", "")).strip(),
        "is_safe": str(record.get("label", "")).lower() == "safe",
        "category": str(record.get("type") or record.get("focus") or "unknown"),
    }


def load_xstest_subset(
    limit: int | None = None,
    label_filter: int | None = None,
    seed: int = 42,
    hf_token: str = "",
) -> list[XSTestExampleRow]:
    """Load XSTest dataset from HuggingFace.

    Args:
        limit: Max samples to load (None = all)
        label_filter: Filter by label (0=safe, 1=unsafe, None=both)
        seed: Random seed
        hf_token: HuggingFace API token

    Returns:
        List of normalized XSTest examples
    """
    ds = load_dataset("Paul/XSTest", split="train", token=hf_token)

    if label_filter is not None:
        ds = ds.filter(lambda x: x["label"] == label_filter)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return [normalize_xstest_record(ds[i], i) for i in range(len(ds))]
