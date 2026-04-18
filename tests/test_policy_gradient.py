"""Lightweight checks for policy-gradient helpers (no GPU required)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from src.runtime.policy_gradient import (
    pad_gen_ids_batch,
    reinforce_update_batch_sgd,
    rejection_sampling_update_sgd,
    rloo_update_batch_sgd,
)


class _ToyLM(nn.Module):
    """Minimal LM that exposes logits for the policy-gradient functions."""

    def __init__(self, vocab: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, 16)
        self.head = nn.Linear(16, vocab)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> object:
        h = self.embed(input_ids)
        logits = self.head(h)
        return type("LMOut", (), {"logits": logits})()


class TestPolicyGradient(unittest.TestCase):
    def test_pad_gen_ids_batch(self) -> None:
        a = torch.tensor([[1, 2, 3]], dtype=torch.long)
        b = torch.tensor([[4, 5]], dtype=torch.long)
        batched, lengths = pad_gen_ids_batch([a, b], pad_id=0)
        self.assertEqual(list(batched.shape), [2, 3])
        self.assertEqual(lengths, [3, 2])

    def test_reinforce_and_rloo_finite_loss(self) -> None:
        torch.manual_seed(0)
        m = _ToyLM()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        # Single row: prompt len 2, completion len 2 -> seq len 4
        gen_ids = torch.tensor([[10, 11, 20, 21]], dtype=torch.long)
        rewards = torch.tensor([1.0], dtype=torch.float32)
        pl = [2]
        # R12: reinforce/rloo now return (loss, logprob_sums, kl_value).
        loss_r, _, kl_r = reinforce_update_batch_sgd(m, opt, gen_ids, pl, rewards, valid_seq_lens=[4])
        self.assertTrue(loss_r == loss_r and abs(loss_r) < 1e6)
        self.assertEqual(kl_r, 0.0)  # no ref_log_probs / kl_coeff -> no KL applied

        gen_ids2 = torch.tensor([[10, 11, 20, 21], [10, 11, 22, 23]], dtype=torch.long)
        rewards2 = torch.tensor([1.0, 0.0], dtype=torch.float32)
        pl2 = [2, 2]
        loss_l, _, kl_l = rloo_update_batch_sgd(m, opt, gen_ids2, pl2, rewards2, valid_seq_lens=[4, 4])
        self.assertTrue(loss_l == loss_l and abs(loss_l) < 1e6)
        self.assertEqual(kl_l, 0.0)

    def test_rejection_sampling_skips_when_no_success(self) -> None:
        m = _ToyLM()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        gen_ids = torch.tensor([[10, 11, 20, 21]], dtype=torch.long)
        rewards = torch.tensor([0.0], dtype=torch.float32)
        loss, n = rejection_sampling_update_sgd(
            m, opt, gen_ids, [2], rewards, valid_seq_lens=[4], success_threshold=0.5
        )
        self.assertIsNone(loss)
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
