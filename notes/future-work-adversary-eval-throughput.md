# Future work: adversary eval throughput & GPU utilization

This note captures why ASR / eval runs in `runs/adversary_run.py` can leave GPUs underutilized, what already exists in code, what can be tuned without refactors, and the main **structural** improvement left: **batched adversary forwards**.

## Pipeline recap (`evaluate_asr`)

Eval for each behavior is **adversary → target → (then aggregate)**. Conceptually:

1. **Adversary (local GPU, Unsloth)** — one completion per behavior, **sequential** in the current implementation (`adversary_output` / `adversary_rewrite_sample`).
2. **Target (often vLLM on a remote GPU)** — HTTP OpenAI-compatible client; when `supports_concurrent_target_inference` is true, target calls are **submitted to a thread pool** so the **next** adversary forward can run while **earlier** target requests are still in flight (see `evaluate_asr` in `runs/adversary_run.py`).
3. **Metric** — `evaluate_outputs`: heuristic is cheap; judge uses `HarmbenchJudgeRuntime.judge`, which is already **batched** over all rows in one forward (see `src/runtime/local_runtimes.py`).

So the **judge** step is not the usual bottleneck for wall time at scale. The recurring limiter is **(1)** plus how much **(2)** can overlap.

## Why “sequential adversary forwards” is the big structural limit

For each behavior you **must** have the adversary’s rewritten text before you can call the target with that text. That ordering is fixed. What is *not* fixed is whether you process **behavior A’s adversary** while still waiting on **behavior B’s target** — that overlap is what **target pipelining + `--target-max-workers`** provides.

What you **cannot** do without new code is run **two adversary completions for two different behaviors on the same LoRA model** at the same time on one GPU, unless you **batch** them into a single forward (or use multi-GPU / multi-process sharding).

So:

- While the **adversary** is generating for behavior *i+1*, the **vLLM** GPU can still be busy on targets for behaviors *i*, *i−1*, … up to the worker cap and server limits.
- While **targets** are running, the **adversary** GPU may be idle if adversary work is short and you have not kept enough targets in flight.

If **adversary latency dominates** (long generations, large batch of eval rows), you spend most wall time on the local GPU **one row at a time**, and the remote GPU may often sit idle — pipelining helps less.

If **target latency dominates** and you crank workers + vLLM concurrency, the **local** adversary GPU may sit idle between rows — that is also expected.

## Speedups that do **not** require adversary batching (already or config-only)

| Lever | Notes |
|--------|--------|
| Target pipelining | Already in `evaluate_asr` when HTTP target supports concurrent inference. |
| `--target-max-workers` | Caps concurrent target HTTP calls; raise toward eval set size / what vLLM can serve. |
| vLLM server settings | e.g. higher concurrent sequences / scheduling so the server can actually run multiple requests (e.g. `--max-num-seqs` and related flags — version-dependent). |
| `--eval-method heuristic` | Skips judge; faster iteration when the softer metric is acceptable. |
| Fewer / shorter tokens | `max_new_tokens`, `target_max_new_tokens` in config — linear effect on generation time. |
| Smaller eval sets | `--csv-test-size` or fewer rows in the CSV. |
| **Sharded eval** | Run multiple processes on disjoint CSV slices (different machines or GPUs), merge CSVs / metrics — parallelizes **adversary** across processes without code changes. |

## Future work: optional **batched adversary** path (larger change)

**Goal:** For eval (and optionally training rollouts), issue **multiple adversary completions in one forward** by extending the stack beyond single-sample `adversary_rewrite_sample` / `adversary_output`.

**Rough scope:**

- **API:** e.g. `adversary_rewrite_batch(prompts: list[str], ...) -> list[dict]` with padding and batching rules aligned with the tokenizer / chat template.
- **Runtime:** `UnslothAdversaryRuntime` (or whatever backs `GenerationSession`) must expose batched generation or a thin wrapper that calls the underlying model with a batch dimension.
- **Call sites:** `evaluate_asr` could batch behaviors in chunks of size `B` (new flag, e.g. `--adversary-eval-batch-size`): for each chunk, run one batched adversary forward, then **fan out** target requests to the existing thread pool (order preserved).
- **Correctness:** Same ordering of `(behavior, adversary_text, target_response)` as today; careful handling of padding, variable lengths, and LoRA adapters.

**Tradeoffs:** higher peak VRAM on the adversary GPU; more complex failure semantics; worth benchmarking vs. sharded multi-process eval.

**When it matters most:** Large eval sets where **adversary** time dominates and you want higher **local** GPU utilization without spinning up multiple processes.

## Related code references

- `runs/adversary_run.py` — `evaluate_asr`, `--target-max-workers`
- `src/run_pipeline.py` — `adversary_rewrite_sample` (single-sample)
- `src/runtime/sessions.py` — `timed_target_generate`, `cap_thread_workers`, `run_target_requests_ordered`
- `src/runtime/openai_http.py` — `supports_concurrent_target_inference`
- `src/runtime/evaluation.py` — `evaluate_outputs`
- `src/runtime/local_runtimes.py` — `HarmbenchJudgeRuntime.judge` (batched)

---

*Added as a future-work item for throughput tuning; not a commitment to implement.*
