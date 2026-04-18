"""Shared adversary policy-gradient steps (REINFORCE, RLOO, rejection sampling).

Used by staged CoEV and adversary-only runners. Tensor shapes: batched ``gen_ids``
of shape ``(batch, seq)`` with per-row prompt prefix lengths and optional valid
lengths before padding.

R12 additions
-------------
``compute_reference_log_probs`` computes frozen-base logprobs by disabling the
active LoRA adapter, and the REINFORCE / RLOO update functions accept an
optional ``ref_log_probs`` plus ``kl_coeff`` to add a per-token KL penalty to
the loss. Rejection sampling is intentionally left untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


def compute_reference_log_probs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    response_start_positions: list[int] | None = None,
) -> torch.Tensor:
    """Compute log-softmax logprobs under the frozen base model (LoRA disabled).

    Returns a tensor shaped ``(B, T-1, V)`` — the same layout as ``log_probs``
    produced inside :func:`rloo_update_batch_sgd` /
    :func:`reinforce_update_batch_sgd` — so callers can reuse the existing
    per-row completion-slice logic (``prompt_len - 1 : prompt_len - 1 + comp_len``)
    to gather sampled-token KL terms.

    ``attention_mask`` and ``response_start_positions`` are accepted for
    future use (e.g. returning a pre-sliced completion-only tensor) but are
    not consumed by the current implementation — the full next-token
    distribution tensor is returned and callers mask to the completion region
    at the gather step.

    # TODO: VERIFY — PEFT disable/enable adapter API for peft==0.18.1.
    # Verified at implementation time: ``PeftModel.disable_adapter_layers``
    # / ``enable_adapter_layers`` are NOT directly exposed on ``PeftModel`` in
    # 0.18.1 (confirmed via hasattr); only the ``disable_adapter()`` context
    # manager is. The context manager is what we use below. If an upgrade
    # moves the method pair onto PeftModel, ``with model.disable_adapter():``
    # still works.
    """
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            with model.disable_adapter():
                out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits
        # Match the in-update shape: next-token distribution at each position t<T.
        ref_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return ref_log_probs


def _per_token_kl_at_sampled_positions(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    gen_ids: torch.Tensor,
    prompt_lens: list[int],
    valid_seq_lens: list[int],
) -> torch.Tensor:
    """Mean per-token KL(policy||reference) at the sampled action token.

    Standard RLHF approximation: gather ``log pi(a_t | s_t) - log ref(a_t | s_t)``
    over the completion region (``prompt_len : valid_end``) and average.

    # TODO: VERIFY — tensor shapes. ``log_probs`` and ``ref_log_probs`` must
    # both be (B, T-1, V) from ``F.log_softmax(logits[:, :-1, :], dim=-1)``.
    # # TODO: VERIFY — masking. We slice positions ``prompt_len - 1 :
    # prompt_len - 1 + comp_len`` inside the per-row gather (same indices as
    # the existing logprob_sums loop), so prompt tokens are excluded by
    # construction and padding tokens are excluded via ``valid_seq_lens``.
    """
    per_row_kls: list[torch.Tensor] = []
    for i in range(gen_ids.size(0)):
        prompt_len = int(prompt_lens[i])
        valid_end = int(valid_seq_lens[i])
        comp_ids = gen_ids[i, prompt_len:valid_end]
        if comp_ids.numel() == 0:
            continue
        start = prompt_len - 1
        end = start + comp_ids.size(0)
        lp_pi = torch.gather(log_probs[i, start:end, :], -1, comp_ids.unsqueeze(-1)).squeeze(-1)
        lp_ref = torch.gather(ref_log_probs[i, start:end, :], -1, comp_ids.unsqueeze(-1)).squeeze(-1)
        per_row_kls.append((lp_pi - lp_ref).mean())
    if not per_row_kls:
        return torch.tensor(0.0, device=gen_ids.device)
    return torch.stack(per_row_kls).mean()


def pad_gen_ids_batch(
    batch_gen_ids: Sequence[torch.Tensor], pad_id: int
) -> tuple[torch.Tensor, list[int]]:
    """Right-pad (1, L) tensors to a common L for batched forward; keep valid lengths."""

    lengths = [int(x.shape[-1]) for x in batch_gen_ids]
    max_len = max(lengths)
    rows: list[torch.Tensor] = []
    for x in batch_gen_ids:
        if x.dim() != 2 or x.size(0) != 1:
            raise ValueError(f"expected gen_ids shape (1, L), got {tuple(x.shape)}")
        L = x.size(1)
        if L < max_len:
            pad = torch.full(
                (1, max_len - L),
                pad_id,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=1)
        rows.append(x)
    return torch.cat(rows, dim=0), lengths


def reinforce_update_batch_sgd(
    model: Any,
    optimizer: Any,
    gen_ids: Any,
    prompt_lens: list[int],
    rewards: Any,
    valid_seq_lens: list[int] | None = None,
    *,
    ref_log_probs: torch.Tensor | None = None,
    kl_coeff: float = 0.0,
) -> tuple[float, Any, float]:
    """REINFORCE-style SGD step: mean over batch of (-reward_i * sum log pi).

    When ``kl_coeff > 0`` and ``ref_log_probs`` is provided, adds
    ``kl_coeff * mean_per_token_KL(policy || reference)`` over the sampled
    tokens. Returns a 3-tuple of ``(loss_val, logprob_sums_detached, kl_val)``;
    ``kl_val`` is ``0.0`` when KL is disabled.
    """

    model.train()
    if valid_seq_lens is None:
        valid_seq_lens = [int(gen_ids.size(1))] * int(gen_ids.size(0))

    out = model(input_ids=gen_ids, use_cache=False)
    logits = out.logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    logprob_sums = []
    for i in range(gen_ids.size(0)):
        prompt_len = int(prompt_lens[i])
        valid_end = int(valid_seq_lens[i])
        comp_ids = gen_ids[i, prompt_len:valid_end].clone()
        if comp_ids.numel() == 0:
            logprob_sums.append(torch.tensor(0.0, device=gen_ids.device))
            continue
        start = prompt_len - 1
        end = start + comp_ids.size(0)
        lp = log_probs[i, start:end, :]
        tok_lp = torch.gather(lp, -1, comp_ids.unsqueeze(-1)).squeeze(-1)
        logprob_sums.append(tok_lp.sum())

    logprob_sums = torch.stack(logprob_sums)
    pg_loss = -(rewards * logprob_sums).mean()

    kl_val = 0.0
    if kl_coeff > 0.0 and ref_log_probs is not None:
        kl_term = _per_token_kl_at_sampled_positions(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            gen_ids=gen_ids,
            prompt_lens=prompt_lens,
            valid_seq_lens=valid_seq_lens,
        )
        loss = pg_loss + kl_coeff * kl_term
        kl_val = float(kl_term.detach().cpu())
    else:
        loss = pg_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu()), logprob_sums.detach(), kl_val


def rloo_update_batch_sgd(
    model: Any,
    optimizer: Any,
    gen_ids: torch.Tensor,
    prompt_lens: list[int],
    rewards: torch.Tensor,
    valid_seq_lens: list[int] | None = None,
    *,
    ref_log_probs: torch.Tensor | None = None,
    kl_coeff: float = 0.0,
) -> tuple[float, Any, float]:
    """RLOO policy-gradient step (leave-one-out baseline over batch rewards).

    When ``kl_coeff > 0`` and ``ref_log_probs`` is provided, adds
    ``kl_coeff * mean_per_token_KL(policy || reference)`` over the sampled
    tokens. Returns a 3-tuple of ``(loss_val, logprob_sums_detached, kl_val)``;
    ``kl_val`` is ``0.0`` when KL is disabled.
    """

    model.train()
    if valid_seq_lens is None:
        valid_seq_lens = [int(gen_ids.size(1))] * int(gen_ids.size(0))

    out = model(input_ids=gen_ids, use_cache=False)
    logits = out.logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    logprob_sums = []
    for i in range(gen_ids.size(0)):
        prompt_len = int(prompt_lens[i])
        valid_end = int(valid_seq_lens[i])
        comp_ids = gen_ids[i, prompt_len:valid_end].clone()
        if comp_ids.numel() == 0:
            logprob_sums.append(torch.tensor(0.0, device=gen_ids.device))
            continue
        start = prompt_len - 1
        end = start + comp_ids.size(0)
        lp = log_probs[i, start:end, :]
        tok_lp = torch.gather(lp, -1, comp_ids.unsqueeze(-1)).squeeze(-1)
        logprob_sums.append(tok_lp.sum())

    logprob_sums = torch.stack(logprob_sums)
    k = rewards.size(0)
    if k > 1:
        mean_reward = rewards.mean()
        advantages = (rewards - mean_reward) * (k / (k - 1))
    else:
        advantages = rewards

    pg_loss = -(advantages.detach() * logprob_sums).mean()

    kl_val = 0.0
    if kl_coeff > 0.0 and ref_log_probs is not None:
        kl_term = _per_token_kl_at_sampled_positions(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            gen_ids=gen_ids,
            prompt_lens=prompt_lens,
            valid_seq_lens=valid_seq_lens,
        )
        loss = pg_loss + kl_coeff * kl_term
        kl_val = float(kl_term.detach().cpu())
    else:
        loss = pg_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu()), logprob_sums.detach(), kl_val


def rejection_sampling_update_sgd(
    model: Any,
    optimizer: Any,
    gen_ids: torch.Tensor,
    prompt_lens: list[int],
    rewards: torch.Tensor,
    valid_seq_lens: list[int] | None = None,
    success_threshold: float = 0.5,
) -> tuple[float | None, int]:
    """Gradient step on successful samples only (reward > success_threshold); else skip."""

    success_mask = rewards > success_threshold
    n_successes = int(success_mask.sum().item())
    if n_successes == 0:
        return None, 0

    idx = success_mask.nonzero(as_tuple=True)[0]
    s_gen_ids = gen_ids[idx]
    s_prompt_lens = [prompt_lens[i] for i in idx.tolist()]
    s_valid_lens = (
        [valid_seq_lens[i] for i in idx.tolist()]
        if valid_seq_lens is not None
        else [int(s_gen_ids.size(1))] * n_successes
    )

    model.train()
    out = model(input_ids=s_gen_ids, use_cache=False)
    log_probs = F.log_softmax(out.logits[:, :-1, :], dim=-1)

    logprob_sums: list[torch.Tensor] = []
    for i in range(s_gen_ids.size(0)):
        pl = s_prompt_lens[i]
        ve = s_valid_lens[i]
        comp_ids = s_gen_ids[i, pl:ve].clone()
        if comp_ids.numel() == 0:
            logprob_sums.append(torch.tensor(0.0, device=s_gen_ids.device))
            continue
        lp = log_probs[i, pl - 1 : pl - 1 + comp_ids.size(0), :]
        logprob_sums.append(torch.gather(lp, -1, comp_ids.unsqueeze(-1)).squeeze(-1).sum())

    loss = -torch.stack(logprob_sums).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu()), n_successes
