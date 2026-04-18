# R12 — KL penalty implementation notes

Concrete implementation guide. No code yet: all references are to file paths
and line numbers in this branch so the diff is small and easy to review.

## Goal

Add a KL-divergence term to the adversary policy-gradient loss that keeps the
trained LoRA policy close to the frozen base model (the LoRA initialization).
Prevents the adversary from collapsing to degenerate outputs and gives a
principled knob to trade ASR vs. fluency.

```
loss = policy_gradient_loss + kl_coeff * mean_per_token_KL(policy || reference)
```

Target controlled by a single CLI flag / YAML key: `kl_coeff` (default 0.0 =
disabled, typical value 0.02–0.1).

## Reference policy snapshot — how and when

The adversary is a PEFT LoRA model built in
`src/runtime/local_runtimes.py::UnslothAdversaryRuntime.__init__` (lines
~170–196). The base Qwen model is frozen and only LoRA adapters are trained.
PEFT lets us compute reference logprobs trivially by **disabling the adapter
on the same model**, so we do not need a second full-model snapshot:

```python
with self.model.disable_adapter():
    ref_logits = self.model(input_ids=gen_ids, use_cache=False).logits
```

This is cheap (one extra forward pass per step, same GPU memory footprint)
and always in sync with the base weights.

Add a helper `reference_log_probs(gen_ids) -> torch.Tensor` on
`UnslothAdversaryRuntime` (below `save_adapters` / `load_adapters`, around
line ~275 after the additions in this branch):

```python
@torch.no_grad()
def reference_log_probs(self, gen_ids: torch.Tensor) -> torch.Tensor:
    """Log-softmax under the base model (LoRA disabled). Shape (B, T-1, V)."""
    was_training = self.model.training
    self.model.eval()
    with self.model.disable_adapter():
        out = self.model(input_ids=gen_ids, use_cache=False)
    logits = out.logits
    # Match layout used by {reinforce, rloo, rejection_sampling}_update_batch_sgd:
    # they slice log_probs[:, :-1, :] (predict next-token distribution at each position).
    ref_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    if was_training:
        self.model.train()
    return ref_log_probs
```

Note: `torch.nn.functional as F` needs to be imported in `local_runtimes.py`
(currently imported as `import torch.nn.functional as f` in that file — reuse
`f` or rename; match the existing convention).

## Where the KL term plugs into the loss

`src/runtime/policy_gradient.py` exposes three update functions. All three
share the same structure; patch each.

### `reinforce_update_batch_sgd` (lines 41–78)

Current body (line numbers from this branch):

- Line 55: `out = model(input_ids=gen_ids, use_cache=False)`
- Line 56: `logits = out.logits`
- Line 57: `log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)`
- Lines 59–71: per-example `logprob_sums` loop gathering `tok_lp`
- Line 73: `logprob_sums = torch.stack(logprob_sums)`
- Line 74: `loss = -(rewards * logprob_sums).mean()` ← insert KL term just before

Add two parameters to the function signature:

```python
def reinforce_update_batch_sgd(
    model, optimizer, gen_ids, prompt_lens, rewards,
    valid_seq_lens=None,
    kl_coeff: float = 0.0,
    ref_log_probs: torch.Tensor | None = None,
) -> tuple[float, Any]:
```

Compute the KL term **after line 57** (we already have `log_probs`) and
**before line 74**:

```python
kl_term = torch.tensor(0.0, device=gen_ids.device)
if kl_coeff > 0.0 and ref_log_probs is not None:
    # Per-token forward KL(policy || reference) = sum_v pi(v) * (log pi(v) - log ref(v))
    # For batched action selection we use the per-token log-prob gap at the
    # sampled token, averaged over valid (non-padded, post-prompt) positions.
    # This is the standard RLHF "per-token KL" approximation.
    kl_diffs: list[torch.Tensor] = []
    for i in range(gen_ids.size(0)):
        prompt_len = int(prompt_lens[i])
        valid_end = int(valid_seq_lens[i]) if valid_seq_lens is not None else int(gen_ids.size(1))
        comp_ids = gen_ids[i, prompt_len:valid_end]
        if comp_ids.numel() == 0:
            continue
        start = prompt_len - 1
        end = start + comp_ids.size(0)
        lp_pi = torch.gather(log_probs[i, start:end, :], -1, comp_ids.unsqueeze(-1)).squeeze(-1)
        lp_ref = torch.gather(ref_log_probs[i, start:end, :], -1, comp_ids.unsqueeze(-1)).squeeze(-1)
        kl_diffs.append((lp_pi - lp_ref).mean())
    if kl_diffs:
        kl_term = torch.stack(kl_diffs).mean()

loss = -(rewards * logprob_sums).mean() + kl_coeff * kl_term
```

Return `float(kl_term.detach().cpu())` alongside the existing loss scalar so
the training log can record it.

### `rloo_update_batch_sgd` (lines 81–125)

Same edit — identical structure. Insert the KL computation between line 113
(`logprob_sums = torch.stack(logprob_sums)`) and line 121
(`loss = -(advantages.detach() * logprob_sums).mean()`). Same two new
parameters, same KL formula.

### `rejection_sampling_update_sgd` (lines 128–172)

Same pattern but only applies KL over the `success_mask`-filtered subset
(`s_gen_ids`, `s_prompt_lens`, `s_valid_lens`). Insert KL after line 166
(`logprob_sums.append(...)` inside the loop collects policy logprobs) and
before line 168 (`loss = -torch.stack(logprob_sums).mean()`).

## CLI plumbing

### `runs/adversary_run.py`

1. **Dataclass field** — `AdversaryTrainingConfig` (lines 74–96 in current
   branch). Add:

   ```python
   kl_coeff: float = 0.0
   ```

2. **argparse** — `parse_args` (lines 491–570). Add after
   `--length-penalty-min-tokens`:

   ```python
   parser.add_argument(
       "--kl-coeff",
       type=float,
       default=run_defaults.get("kl_coeff", 0.0),
       help="KL penalty coefficient (0.0 disables). Typical range 0.02–0.1.",
   )
   ```

3. **Config hydration** — `main()` (lines 591–656). Add to the
   `AdversaryTrainingConfig(...)` constructor:

   ```python
   kl_coeff=float(run_defaults.get("kl_coeff", 0.0)),
   ```

4. **Training loop** — `main()` iteration body (lines 663–824). After the
   existing rollout/reward collection, before the three
   `*_update_batch_sgd` calls (lines 750–778):

   ```python
   ref_log_probs = None
   if args.kl_coeff > 0.0:
       ref_log_probs = model.reference_log_probs(padded_ids)
   ```

   Then pass `kl_coeff=args.kl_coeff, ref_log_probs=ref_log_probs` to each of
   the three branches (RS / RLOO / REINFORCE).

5. **Training CSV** — `training_rows.append({...})` (lines 803–824). Add
   `"kl_term": kl_term_value` so you can plot KL over time.

### YAML config delta

`configs/r12_kl_penalty.yaml` would inherit from
`configs/r11_rloo_length_penalty.yaml` and only override:

```yaml
runs:
  adversary:
    results_dir: "results/r12_kl_penalty"
    kl_coeff: 0.05
    # length penalty can stay on or be turned off to isolate the KL effect.
    length_penalty_weight: 0.0

scripts:
  unified_runner:
    save_dir: "checkpoints"
    adversary_results_dir: "results/r12_kl_penalty"
```

## Testing

1. Smoke: set `kl_coeff=0.0` → loss/behaviour identical to current branch.
2. Small run with `kl_coeff=0.1` → `kl_term` column should appear in the
   training CSV and trend upward during training (policy diverges from base).
3. Verify `model.disable_adapter()` actually zeros the LoRA contribution —
   quick unit test: same `input_ids` under adapter-enabled vs. adapter-disabled
   context managers should produce different logits iff the adapter has
   been trained for at least one step.

## Known risks / TODO

- `model.disable_adapter()` is a PEFT context manager; in rare PEFT versions
  without it, the fallback is to deep-copy the base model before training and
  keep a frozen reference session. Check `peft.__version__`; if it is
  pre-0.5 upgrade before relying on `disable_adapter`.
- Extra forward pass per step roughly doubles the per-step wall-clock for the
  adversary update when KL is active; factor that into the run budget.
- The per-token-at-sampled-token KL estimator above is the standard RLHF
  approximation (used by TRL / DeepSpeed-Chat). If you need full forward-KL
  over the whole vocab, switch to `F.kl_div(log_probs, ref_log_probs.exp())`
  — cost is the same forward pass, but memory is ~vocab_size higher per
  token.
