"""Shared adversary policy-gradient steps (REINFORCE, RLOO, rejection sampling).

Used by staged CoEV and adversary-only runners. Tensor shapes: batched ``gen_ids``
of shape ``(batch, seq)`` with per-row prompt prefix lengths and optional valid
lengths before padding.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


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
) -> tuple[float, Any]:
    """REINFORCE-style SGD step: mean over batch of (-reward_i * sum log pi)."""

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
    loss = -(rewards * logprob_sums).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu()), logprob_sums.detach()


def rloo_update_batch_sgd(
    model: Any,
    optimizer: Any,
    gen_ids: torch.Tensor,
    prompt_lens: list[int],
    rewards: torch.Tensor,
    valid_seq_lens: list[int] | None = None,
    ref_log_prob_sums: torch.Tensor | None = None,
    kl_coeff: float = 0.0,
) -> tuple[float, Any]:
    """RLOO policy-gradient step (leave-one-out baseline over batch rewards)."""

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

    loss = -(advantages.detach() * logprob_sums).mean()
    if ref_log_prob_sums is not None and kl_coeff > 0.0:
        kl_per_sample = logprob_sums - ref_log_prob_sums.to(logprob_sums.device)
        loss = loss + kl_coeff * kl_per_sample.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu()), logprob_sums.detach()


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
