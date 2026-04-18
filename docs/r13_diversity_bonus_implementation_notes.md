# R13 — diversity bonus implementation notes

Concrete implementation guide for adding an embedding-based diversity bonus
to the adversary training loop, alongside the existing length penalty. No
code yet; file paths and line numbers reference this branch.

## Goal

Penalise rollout batches where the K adversary rewrites are semantically
similar to each other (embedding-cosine similarity), so the policy is
discouraged from mode-collapsing onto a single "winning" paraphrase. The
bonus is added to the per-sample reward before the policy-gradient step,
parallel to how `shape_reward_with_length_penalty` already shapes reward
in `runs/adversary_run.py::shape_reward_with_length_penalty` (lines
163–182 in this branch).

```
shaped_reward_i = (1 - w_div) * base_reward_i + w_div * diversity_i
```

where `diversity_i ∈ [0, 1]` is `1 - mean_cosine_similarity(embedding_i, other_embeddings)`.

`w_div = 0.0` disables the bonus (backward compatible).

## New module — `src/runtime/diversity.py`

Create a small module encapsulating the sentence-transformer embedding and
the batch-level similarity reduction:

```python
"""Embedding-based diversity bonus for adversary rollouts.

Loads a small sentence-transformer model lazily (first call) and exposes a
pure helper that turns a list of K adversarial rewrite strings into a
per-sample diversity vector in [0, 1], suitable for blending into reward.
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
import torch

_MODEL_LOCK = threading.Lock()
_MODEL: object | None = None


def _load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load the sentence-transformer on first use.

    all-MiniLM-L6-v2 is ~80MB, CPU-inferable in <10ms per sentence, and is
    the standard compromise between quality and footprint. Overridable with
    the DIVERSITY_EMBED_MODEL env var for experimentation.
    """
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            import os
            from sentence_transformers import SentenceTransformer  # noqa: WPS433

            effective = os.environ.get("DIVERSITY_EMBED_MODEL", model_name)
            # GPU is fine if available — the model is tiny — but default to CPU
            # because the big adversary model + vLLM already saturate the H100.
            _MODEL = SentenceTransformer(effective, device="cpu")
    return _MODEL


def batch_diversity_scores(texts: Sequence[str]) -> np.ndarray:
    """Return per-sample diversity in [0, 1] for a batch of K strings.

    diversity_i = 1 - mean_{j != i} cosine_sim(embed_i, embed_j)
    diversity_i = 1.0 when K <= 1 (nothing to compare against).
    """
    k = len(texts)
    if k <= 1:
        return np.ones(k, dtype=np.float32)
    model = _load_model()
    emb = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    sim = emb @ emb.T  # (K, K), cosine since embeddings are L2-normalized
    np.fill_diagonal(sim, 0.0)
    mean_sim = sim.sum(axis=1) / (k - 1)
    return np.clip(1.0 - mean_sim, 0.0, 1.0).astype(np.float32)
```

## `pyproject.toml` dependency

Add `sentence-transformers>=2.7` to the project dependencies. `uv sync`
during `scripts/launch_unified_prime.sh` will pick it up automatically on
first run.

## CLI plumbing

### `runs/adversary_run.py`

1. **Dataclass field** — `AdversaryTrainingConfig` (lines 74–96). Add:

   ```python
   diversity_weight: float = 0.0
   diversity_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
   ```

2. **argparse** — `parse_args` (lines 491–570). Add:

   ```python
   parser.add_argument(
       "--diversity-weight",
       type=float,
       default=run_defaults.get("diversity_weight", 0.0),
       help="Weight for embedding-based diversity reward shaping (0.0 disables).",
   )
   parser.add_argument(
       "--diversity-model-name",
       default=run_defaults.get(
           "diversity_model_name", "sentence-transformers/all-MiniLM-L6-v2"
       ),
       help="Sentence-transformer model for diversity embeddings.",
   )
   ```

3. **Config hydration** — `main()` (lines 600–614). Add:

   ```python
   diversity_weight=float(run_defaults.get("diversity_weight", 0.0)),
   diversity_model_name=str(run_defaults.get(
       "diversity_model_name", "sentence-transformers/all-MiniLM-L6-v2"
   )),
   ```

### Where to compute the bonus (training loop)

The batched rollout collection lives in the `if args.mode == "train":` block
(lines 663–824). Both the `use_rs` branch (lines 677–709) and the
`else`/REINFORCE-RLOO branch (lines 710–738) build three parallel lists:
`batch_gen_ids`, `batch_rewards`, `adv_texts`. Insert the diversity
shaping **after** both branches finish collecting but **before** the
`pad_gen_ids_batch` / reward tensorization on lines 740–748.

```python
# --- After per-sample length penalty is already baked into batch_rewards ---
if train_cfg.diversity_weight > 0.0 and len(adv_texts) > 1:
    from src.runtime.diversity import batch_diversity_scores
    div_scores = batch_diversity_scores(adv_texts)  # np.float32, shape (K,)
    w = train_cfg.diversity_weight
    batch_rewards = [
        (1.0 - w) * base + w * float(div)
        for base, div in zip(batch_rewards, div_scores)
    ]
```

Place this block immediately **before** line 740
(`tok = model.tokenizer`) so `rewards_t` (line 748) picks up the shaped
values.

### Training log

Append the diversity scores to `training_rows.append({...})` on lines
803–824 (next to the existing `length_penalty_weight` /
`mean_completion_tokens` entries):

```python
"diversity_weight": train_cfg.diversity_weight,
"mean_diversity_score": float(div_scores.mean()) if train_cfg.diversity_weight > 0.0 else "",
```

## Interaction with length penalty

The length penalty (already implemented: `shape_reward_with_length_penalty`
at lines 163–182) is applied **per-sample, before** the diversity step and
updates each reward individually. The diversity bonus is **per-batch**. The
two are compositional:

```
reward_i
  = (1 - w_len) * base_i          + w_len * length_ratio_i      # per-sample
reward_i
  = (1 - w_div) * reward_i        + w_div * diversity_i         # per-batch
```

Enable both with non-zero weights or either independently. Typical
starting values: `length_penalty_weight=0.2, diversity_weight=0.1`.

## YAML config delta

`configs/r13_diversity_bonus.yaml` inherits from R11 and adds:

```yaml
runs:
  adversary:
    results_dir: "results/r13_diversity_bonus"
    diversity_weight: 0.1
    diversity_model_name: "sentence-transformers/all-MiniLM-L6-v2"
    # Keep R11's length penalty on to study the combined effect, or set
    # length_penalty_weight: 0.0 to ablate to diversity-only shaping.
    length_penalty_weight: 0.2
    length_penalty_min_tokens: 50

scripts:
  unified_runner:
    save_dir: "checkpoints"
    adversary_results_dir: "results/r13_diversity_bonus"
```

## Testing

1. Smoke: `diversity_weight=0.0` → `batch_rewards` list identical to current
   branch (no-op path).
2. Unit test `batch_diversity_scores`:
   - K=1 → array of length 1, value 1.0
   - K=4 with identical strings → near-zero diversity (high cosine similarity
     between embeddings → `1 - mean_sim ≈ 0`).
   - K=4 with very different strings → diversity near 1.0.
3. Small training run with `diversity_weight=0.3` → `mean_diversity_score`
   column in training log should trend upward over iterations as the
   policy learns to emit varied rewrites.

## Known risks / TODO

- First call to `_load_model` downloads ~80MB from HuggingFace. Pre-pull in
  `setup_env.sh` to avoid a surprise stall mid-training.
- `sentence-transformers` pulls in `transformers` which is already a
  dependency, but pinning is brittle — if `uv sync` complains about
  incompatible versions, pin `sentence-transformers==2.7.0` explicitly.
- Embedding-based diversity measures **surface-form** diversity. If you want
  to reward **attack-strategy** diversity (e.g. fictional-framing vs.
  decomposition), swap in a classifier over rewrite styles instead — the
  wiring point is identical, only `batch_diversity_scores` changes.
- Encoding happens on CPU by default. If the adversary batch gets large
  (K ≥ 16) this becomes meaningful overhead; switch `device="cuda"` after
  confirming there is free VRAM after the main adversary + judge + vLLM load.
