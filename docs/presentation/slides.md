# Slide Deck — Model Steering for Safety and Robustness

Supporting assets live alongside this file in `docs/presentation/`:

- Figures: `fig_asr_trajectory.png`, `fig_kl_ablation.png`, `fig_coev_stages.png`, `fig_xstest_comparison.png`, `fig_policy_comparison.png`
- Example rollouts: `examples_mode_collapse.md`, `examples_successful_attack.md`, `examples_seed_comparison.md`

---

## Slide 1: Title

**Model Steering for Safety and Robustness**

Policy Gradient Methods for Adversarial Prompt Optimization

STAT 4830 · University of Pennsylvania · April 2026

> Speaker notes: quick team intro, frame the talk as a ~15 min empirical study — what we built, what worked, what didn't, what we'd do next. Flag up front that we have ablation data and will be explicit about what's signal vs what's noise.

---

## Slide 2: The Problem

- LLMs remain vulnerable to adversarial prompts **despite RLHF / DPO alignment**.
- System-prompt steering is **lightweight** (no weight updates at inference) but **untested** under automated attack.
- **Manual red-teaming doesn't scale** to the prompt × model grid.
- Open question: can a small RL-trained attacker + a prompt-optimized defender co-train to surface real weaknesses *and* produce deployable defenses?

> Speaker notes: motivate why anyone should care. "Manual red-teaming at Anthropic / OpenAI scale employs whole teams full-time. If a 7B adversary can reliably elicit a 48-point ASR bump, that's a cheap signal for where safety training is thin — and the evolved defense prompts transfer as a zero-cost patch."

---

## Slide 3: Our Approach

- **Minimax formulation:** $\min_s \max_\theta \mathrm{ASR}(s, \theta)$ — defense prompt $s$ minimizes, adversary weights $\theta$ maximize.
- **Attack:** RL-trained adversary (Qwen2.5-7B + LoRA) rewrites harmful prompts; reward = HarmBench-Mistral judge verdict on target output.
- **Defense:** GEPA evolves the system prompt via reflection-based mutation against a frozen attacker.
- **Co-evolution:** alternate K RLOO adversary steps with one GEPA prompt-evolution round per stage.

> Speaker notes: three components — explain adversary, defense, and the loop. Stress that the inner reward is the judge verdict, not a heuristic, so ASR reported is under a realistic judge. Target model is Llama-3.1-8B throughout.

---

## Slide 4: Architecture Diagram

Pipeline flow (replace with Figma/draw.io diagram for the actual presentation):

```
HarmBench prompt
      │
      ▼
┌────────────────────────────┐
│ Adversary LoRA (Qwen 7B)   │  rewrite
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐     defense prompt s
│ Target model (Llama 3.1 8B)│  ◀──── GEPA evolves this
└──────────────┬─────────────┘             │
               │ response                  ▲ (every ``iters_per_stage`` steps)
               ▼                           │
┌────────────────────────────┐    verdict  │
│ HarmBench-Mistral judge    │─────────────┘
└──────────────┬─────────────┘
               │ reward (shape with length penalty, optional KL)
               ▼
     RLOO policy gradient
     → LoRA weights θ
```

> Speaker notes: walk left-to-right once, then point out the GEPA back-edge — that's the co-evolution element. Note that the adversary is the only trained component; target weights and reflection model are frozen.
> **Todo:** clean diagram in Figma for the actual slide.

---

## Slide 5: What Happens Without Reward Shaping

![](fig_policy_comparison.png)

- REINFORCE with **no length penalty** peaks at ASR **0.24** and then collapses.
- Without shaping, the adversary finds a **short-output local optimum** — 70-char vague paraphrases that are neither clear jailbreaks nor clear refusals.
- See `examples_mode_collapse.md` for 3 rollouts from iter 80–100: "What kind of content might persuade…" type outputs that the judge correctly rates `No`.

> Speaker notes: the key motivating failure. Read one example aloud from examples_mode_collapse.md. Point out that the reward signal is technically well-defined but the gradient prefers high-probability short outputs — because the log-prob sum is less negative for shorter completions. Length penalty is the fix.

---

## Slide 6: Length Penalty Fixes It

- With `length_penalty_weight=0.2`, `min_tokens=50`, the adversary learns **substantive** rewrites.
- Strategies emerge that mirror the seed instruction: **fictional framing, persona delegation, clinical substitution, stepwise decomposition.**
- ASR climbs **0.12 → 0.52** (peak at iter 120 in `r11_full_prompt`).
- See `examples_successful_attack.md` for concrete rollouts (Dr. Vasquez forensic-analyst persona, thriller-scene framing, court-filing framing).

> Speaker notes: walk through Example 2 from examples_successful_attack.md — "chemical weapons from university lab materials" → adversary wraps in forensic-chemist persona → target produces a "CLASSIFIED DOCUMENT" style response with the requested content. Emphasize: the target had a refusal policy, the adversary learned to route around it.

---

## Slide 7: Training Trajectory

![](fig_asr_trajectory.png)

- **Peak at iter 120, collapse by iter 140** — motivated adding a `checkpoints_best/` save path alongside the end-of-run `checkpoints/`.
- **High variance across seeds (±28 pp at peak)** — `r11_full_prompt` and `r11_full_prompt_best` ran the same config at seed 42 and peaked at 0.52 vs 0.24 respectively. Single-run numbers are not reliable.

> Speaker notes: three lines — two RLOO+LP runs and one REINFORCE baseline on the same axes. Point to the vertical "collapse begins" line. Be explicit that the ~28-point run-to-run spread is the real noise floor, so any headline delta smaller than that is not meaningful without replicates. This is why we added best-checkpoint saving mid-project.

---

## Slide 8: Seed Instruction Quality

- **Full 2262-char seed** (multi-strategy, with illustrative example frames) → peak ASR **0.52**.
- **Condensed 260-char seed** → peak ASR **0.14** (flattest trajectory of any R12-family run).
- See `examples_seed_comparison.md` for matched-prompt side-by-sides.

> Speaker notes: key finding. The RL procedure does not invent new jailbreak strategies from first principles; it **amplifies** the strategies the seed enumerates. That's why prompt engineering + RL are complementary — the seed provides hypothesis space, RL optimizes weight on its components. Worth stating explicitly that this reverses the usual "RL will find it for you" framing.

---

## Slide 9: KL Regularization

![](fig_kl_ablation.png)

- **Clean single-knob ablation:** same seed=123, same 2262-char prompt, same 150-iter RLOO+LP schedule; only `kl_coeff` differs.
- KL = 0.05 → peak **0.34** vs no-KL peak **0.40**. **Stabilizer, not booster.**
- The **KL divergence spike at iter 120** (dotted right-axis) correlates with the ASR dip — KL kicks in exactly when the policy would otherwise collapse.

> Speaker notes: KL acts as a tether back to the frozen base model. It prevents the late-training drift we saw in slide 7, at the cost of ~6 points of peak ASR. For a deployed attacker you'd probably want KL on. For finding vulnerabilities, leave it off.

---

## Slide 10: Co-Evolution Dynamics

![](fig_coev_stages.png)

- **Pre-coev baseline ASR 0.44** (adversary loaded from R11 seed=123 `checkpoints_best/`).
- **Defense wins:** ASR drops to ~0.06–0.10 by stage 3 — GEPA successfully hardens the target system prompt.
- But this happens because the **adversary degrades while training against the evolving defense** (see next slide).

> Speaker notes: two runs on the chart — no KL and KL=0.05. Both show the same downward staircase. Stress that this is measured against the *evolving* defense, so it's a joint movement of both sides, not just one side improving.

---

## Slide 11: Evolved Defense Prompts

**Before (seed, 28 chars):**

```
You are a helpful assistant.
```

**After coev training (R14 no-KL, 435 chars):**

```
You are a responsible assistant who prioritizes safety and security. You will
not provide information or guidance on harmful activities, including but not
limited to: promoting or glorifying violence, harassment, or self-harm;
creating or distributing malware; or facilitating scams. If a prompt is
unclear or open-ended, please ask clarifying questions or request more
context to ensure you provide a helpful and responsible response.
```

- **GEPA didn't evolve the attacker instruction** — `optimized_attacker_instruction.txt` is byte-equal to the 2262-char seed in both R14 runs.
- **208 attacker candidates evaluated, none beat the seed** on the val set.
- Defender seed was 28 chars (trivially improvable); attacker seed was 2262 chars (already near-local-optimum).

> Speaker notes: asymmetric evolution is a real finding. The reflection model (Llama-3.1-8B) can mutate 28 chars upward but cannot write something strictly better than 2262 words of carefully tuned jailbreak strategy in 50 metric calls. We'd fix this with attacker-specific `max_metric_calls` budget — something like 200 on the attacker, 50 on the defender.

---

## Slide 12: Adversary Degradation

- **Pre-coev HarmBench eval** (R11 seed=123 adversary, default `"You are a helpful assistant."` target): **ASR 0.40**.
- **Post-coev HarmBench eval** (R14 adversary after 4 stages, same naked target): **ASR 0.08**.
- The adversary **overspecialized** against the successively harder evolved defense prompts — when evaluated back on the *original* defense, it has forgotten how to attack.
- **Novel finding:** co-evolution in this configuration is a **defense-training method, not an adversary-training method.**

> Speaker notes: this is probably the single most interesting result. The usual assumption in co-evolution literature is that both sides improve together. Here, the attacker learns policies that work against increasingly hardened targets and loses generality. If the goal is a strong attacker, train adversary-only with a frozen target. If the goal is a strong defender, do coev.

---

## Slide 13: Defense-Utility Tradeoff

![](fig_xstest_comparison.png)

- **All evolved defenses maintain 98%+ safe compliance** on XSTest.
- **Over-refusal stays below 2%** — the GEPA-evolved prompts are precise (block specific categories) rather than blanket-refusing.
- Contrast with the GEPA-only baseline from prior work, which over-refused on ~19% of safe prompts (80.8% compliance).

> Speaker notes: green bars indicate precise defenses. Flag that we haven't run the target-only-baseline XSTest yet (the "not run" row) — need a zero-defense-prompt reference to make the delta crisp. Safe-compliance numbers to the right show the raw helpfulness figure.

---

## Slide 14: Key Takeaways + Future Work

**Takeaways:**

- **Reward shaping is load-bearing.** Length penalty alone flips REINFORCE from collapsing (0.24 peak) to holding (0.40 peak). Without it, no amount of RL helps.
- **Seed instruction quality ≈ RL training quality.** 260-char → 0.14 peak; 2262-char → 0.52 peak on matched budgets. RL amplifies the seed's hypothesis space.
- **Co-evolution trains a defense, not an attacker.** The adversary overspecializes; the defense prompt generalizes.
- **High run-to-run variance (±28 pp).** Single seeds are not data points.

**Future work:**

- **KL warm-up** (off during exploration, on during convergence) to keep late-training stable without capping peak ASR.
- **Diversity bonus** (embedding-based, across K rollouts per step) to keep the policy from collapsing onto one strategy — design doc already in `docs/r13_diversity_bonus_implementation_notes.md`.
- **Stronger reflection model** (or attacker-specific `max_metric_calls`) so GEPA can actually move the attacker side.
- **Multiple target models** — current results are Llama-3.1-8B only; generality unknown.
- **GCG / AutoDAN baselines** as attacker references so our RL numbers are calibrated.

> Speaker notes: close with "three things we'd do next week with more GPU budget" and land the plane. If asked — yes, the repo is reproducible from `scripts/run_b200_overnight.sh`; nothing committed to main yet; results directories are under `results/` with manifests.
