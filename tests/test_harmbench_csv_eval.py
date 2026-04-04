"""Tests for HarmBench behaviors CSV loading and val/test splits."""

from __future__ import annotations

import os
import tempfile

import pandas as pd

from src.data import harmbench_csv_row_to_prompt, load_harmbench_csv_val_test_splits


def test_harmbench_csv_row_to_prompt_with_context() -> None:
    row = pd.Series(
        {
            "Behavior": "Instruction text",
            "ContextString": "Extra\ncontext",
        }
    )
    assert harmbench_csv_row_to_prompt(row) == "Instruction text\n\nExtra\ncontext"


def test_harmbench_csv_row_to_prompt_empty_context() -> None:
    row = pd.Series({"Behavior": "Only behavior", "ContextString": ""})
    assert harmbench_csv_row_to_prompt(row) == "Only behavior"


def test_load_harmbench_csv_val_test_splits_disjoint() -> None:
    lines = [
        "Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID",
    ]
    for i in range(10):
        lines.append(f"b{i},standard,x,,,id{i}")
    content = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    try:
        val, test = load_harmbench_csv_val_test_splits(path, val_size=2, test_size=3, seed=42)
        assert len(val) == 2
        assert len(test) == 3
        ids_val = {x["id"] for x in val}
        ids_test = {x["id"] for x in test}
        assert ids_val.isdisjoint(ids_test)
        assert all("prompt" in x and x["prompt"].startswith("b") for x in val + test)
    finally:
        os.unlink(path)
