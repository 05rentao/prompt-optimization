"""XSTest dataset loader.

XSTest (Röttger et al., 2023) is an overrefusal benchmark.
It contains 250 *safe* prompts that a model should comply with, and
250 *contrast unsafe* prompts that a model should refuse.

A low HarmBench ASR may indicate a model is over-refusing (refusing safe
requests as if they were harmful). XSTest lets you measure that directly:

  - On **safe** prompts  → high refusal rate  = BAD (overrefusal)
  - On **unsafe** prompts → low  refusal rate  = BAD (under-refusal / ASR)

HuggingFace dataset: ``paul-rottger/xstest``

Columns (v2):
  id_v2, type, prompt, focus, note, label  (label ∈ {"safe","unsafe"})

The ``type`` column names the kind of overrefusal test, e.g.::

  safe_contexts        safe_descriptions   safe_homonyms
  safe_figurative      safe_real_group     safe_public_figure
  safe_privacy_public  safe_requested      safe_targets
  contrast_homonyms    contrast_targets    … etc.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

XSTEST_HF_REPO = "paul-rottger/xstest"
XSTEST_SPLIT = "test"

SAFE_LABEL = "safe"
UNSAFE_LABEL = "unsafe"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class XSTestLoader:
    """Load and slice the XSTest overrefusal benchmark."""

    # ------------------------------------------------------------------
    # Main loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        label_filter: str | None = None,
        limit: int | None = None,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Return a list of normalised records.

        Parameters
        ----------
        label_filter:
            ``"safe"`` → only safe prompts (overrefusal probes).
            ``"unsafe"`` → only contrast-unsafe prompts.
            ``None`` → return all 500 prompts.
        limit:
            Optional cap on number of records after filtering.
        seed:
            Random seed used when ``limit`` triggers a shuffle-and-slice.

        Returns
        -------
        List of dicts with keys:
            ``id``, ``prompt``, ``type``, ``focus``, ``label``,
            ``is_harmful_request``
        """
        df = XSTestLoader.load_df(label_filter=label_filter, limit=limit, seed=seed)
        return df.to_dict(orient="records")

    @staticmethod
    def load_df(
        label_filter: str | None = None,
        limit: int | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Return a normalised ``pd.DataFrame``."""
        try:
            ds = load_dataset(XSTEST_HF_REPO, split=XSTEST_SPLIT)
            df = ds.to_pandas()
        except Exception as exc:
            raise RuntimeError(
                f"Could not load XSTest from Hugging Face ({XSTEST_HF_REPO}). "
                "Check your internet connection. "
                f"Original error: {exc}"
            ) from exc

        df = _normalise_columns(df)

        if label_filter is not None:
            if label_filter not in (SAFE_LABEL, UNSAFE_LABEL):
                raise ValueError(
                    f"label_filter must be 'safe', 'unsafe', or None; got {label_filter!r}"
                )
            df = df[df["label"] == label_filter].reset_index(drop=True)

        if limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Convenience splits
    # ------------------------------------------------------------------

    @staticmethod
    def load_safe(limit: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
        """Return only safe prompts (overrefusal probes)."""
        return XSTestLoader.load(label_filter=SAFE_LABEL, limit=limit, seed=seed)

    @staticmethod
    def load_unsafe(limit: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
        """Return only contrast-unsafe prompts."""
        return XSTestLoader.load(label_filter=UNSAFE_LABEL, limit=limit, seed=seed)

    @staticmethod
    def load_split(
        safe_size: int = 250,
        unsafe_size: int = 250,
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (safe_records, unsafe_records) in one call."""
        safe = XSTestLoader.load_safe(limit=safe_size, seed=seed)
        unsafe = XSTestLoader.load_unsafe(limit=unsafe_size, seed=seed)
        return safe, unsafe

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def type_counts(label_filter: str | None = None) -> dict[str, int]:
        """Return ``{type_name: count}`` so you can see the distribution."""
        df = XSTestLoader.load_df(label_filter=label_filter)
        return df["type"].value_counts().to_dict()

    @staticmethod
    def save_csv(path: str = "data/xstest.csv") -> None:
        """Download XSTest and cache it locally as a CSV."""
        df = XSTestLoader.load_df()
        df.to_csv(path, index=False)
        print(f"Saved XSTest ({len(df)} rows) to {path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename / backfill columns so the rest of the codebase gets a uniform schema.

    Output columns guaranteed:
        id, prompt, type, focus, label, is_harmful_request
    """
    # Rename id column (v1 uses 'id', v2 uses 'id_v2')
    if "id_v2" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"id_v2": "id"})

    # Synthesise id if still missing
    if "id" not in df.columns:
        df["id"] = [f"xs_{i}" for i in range(len(df))]

    # Ensure required text columns exist
    if "prompt" not in df.columns:
        # Try common alternatives
        for alt in ("text", "question", "input"):
            if alt in df.columns:
                df = df.rename(columns={alt: "prompt"})
                break
        else:
            raise ValueError(
                f"Cannot find a prompt column in XSTest. Columns: {df.columns.tolist()}"
            )

    for col in ("type", "focus", "label"):
        if col not in df.columns:
            df[col] = ""

    # Derive is_harmful_request for compatibility with HarmBench-style code
    df["is_harmful_request"] = df["label"].str.lower() == UNSAFE_LABEL

    # Cast id to string
    df["id"] = df["id"].astype(str)

    # Keep a clean column order
    keep = ["id", "prompt", "type", "focus", "label", "is_harmful_request"]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    filter_arg = sys.argv[1] if len(sys.argv) > 1 else None
    df = XSTestLoader.load_df(label_filter=filter_arg)
    print(f"Loaded {len(df)} XSTest records (filter={filter_arg!r})")
    print(df[["id", "label", "type", "prompt"]].head(10).to_string(index=False))
    print("\nType distribution:")
    for t, n in XSTestLoader.type_counts(label_filter=filter_arg).items():
        print(f"  {t}: {n}")
