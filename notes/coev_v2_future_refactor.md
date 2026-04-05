# CoEV v2 — modularization status

`runs/coev_v2_run.py` merges the former RLOO-only CoEV script and adds rejection sampling, multi-query rewards, and adversary prompt presets.

**Done:** Shared policy-gradient helpers live in [`src/runtime/policy_gradient.py`](../src/runtime/policy_gradient.py): `pad_gen_ids_batch`, `reinforce_update_batch_sgd`, `rloo_update_batch_sgd`, `rejection_sampling_update_sgd`. [`runs/coev_v2_run.py`](../runs/coev_v2_run.py) and [`runs/adversary_run.py`](../runs/adversary_run.py) import from this module. Lightweight checks: [`tests/test_policy_gradient.py`](../tests/test_policy_gradient.py).

**Still optional (follow-up):**

1. Extract `multi_query_reward` next to `src/run_pipeline` evaluation utilities, shared by CoEV v2 and future runners.
2. Re-import from `policy_gradient` in legacy [`runs/coev_run.py`](../runs/coev_run.py) when that script is retired or slimmed down.

This keeps tensor math in one place and makes drift less likely.
