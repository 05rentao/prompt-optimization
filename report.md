# Prompt Optimization Adversarial Attacks on GEPA-Aided Models

## Problem Statement

### What are we optimizing?
This project studies **adversarial prompt optimization** against **GEPA (Generalized Prompt Optimization Algorithm)**-aided language models. Specifically, we aim to *numerically optimize an adversary model* that takes a benign user prompt as input and rewrites it into an adversarial prompt such that, **even when combined with a GEPA-generated system prompt**, the downstream language model produces a **bad output** (non-optimal, incorrect, or toxic).

Formally, the optimization target is the adversary model’s parameters, which maximize the probability that the GEPA-protected system produces a failure case:

> (system_prompt from GEPA, rewritten_prompt from adversary) → bad answer

### Why does this problem matter?
GEPA is designed to improve robustness and alignment by learning a system prompt that ensures acceptable outputs across a distribution of prompts. If an adversary can systematically bypass GEPA through optimized prompt rewriting, it exposes **fundamental weaknesses in prompt-based alignment and safety mechanisms**. This has direct implications for:
- Safety evaluation of LLM alignment techniques
- Robustness of prompt optimization methods
- Understanding adversarial failure modes in deployed systems

From a numerical optimization perspective, this project reframes *prompt hacking* as a **reinforcement learning optimization problem over discrete text outputs**, making it a compelling application of optimization techniques in modern ML systems.

### How will we measure success?
Success is measured by:
- **Attack Success Rate (ASR):** fraction of prompts where the adversary causes GEPA to produce a bad output
- **Reward convergence:** stability and improvement of expected reward during adversary training
- **Qualitative metric of toxicity:** TBD: \{ Toxicity-Bert, Toxcity Library\}

### Constraints
- Black-box access to GEPA outputs (no gradient access)
- Discrete text generation space
- Limited compute budget (single-GPU training)
- Use of open-source models only for the adversary
- Which model to use as student/teacher for GEPA
- Limited API calls
- API safety
- Existing models are already too safe

### Data requirements
- A small-to-medium **prompt–answer dataset** used to train GEPA

### What could go wrong?
- Reward sparsity causing unstable REINFORCE training
- Adversary learns trivial prompt corruption instead of subtle attacks
- GEPA system prompt overfits to training data
- Ambiguity in defining “bad” answers

---

## Technical Approach

### Mathematical formulation
Let:
- \(x\) be the original user prompt
- \(a_\theta(x)\) be the adversary’s rewritten prompt parameterized by \(\theta\)
- \(s = \text{GEPA}(D)\) be the learned system prompt from training data \(D\)
- \(y = M(s, a_\theta(x))\) be the model output

Define reward:

\[
R(y) = \begin{cases}
+1 & \text{if } y \text{ is a bad output} \\
-1 & \text{otherwise}
\end{cases}
\]

Objective:

\[
\max_{\theta} \; \mathbb{E}_{x \sim \mathcal{X}} [ R(M(s, a_\theta(x))) ]
\]

This is optimized using **REINFORCE**, since gradients cannot pass through GEPA or the downstream LLM.

### Algorithm choice and justification
- **GEPA** is used to generate a robust system prompt from (prompt, answer) pairs
- **Adversary model:** small instruction-tuned LLM (e.g., Qwen, Gemma) trained with RL
- **REINFORCE** is chosen due to:
  - Discrete action space (text)
  - Black-box reward signal
  - Conceptual simplicity for an initial project

### PyTorch implementation strategy
- Use **Unsloth** for efficient fine-tuning and RL training
- Freeze GEPA and target model weights
- Adversary outputs rewritten prompts token-by-token
- Reward computed from downstream model output using heuristic or classifier
- Policy gradient update with baseline for variance reduction

### Validation methods
- Held-out prompt set for attack evaluation
- Manual inspection of adversarial prompts
- Comparison against random or heuristic prompt perturbations

### Resource requirements
- 1 GPU (≈24GB VRAM preferred)
- Small batch sizes due to long context lengths
- Training time: hours, not days

---

## Initial Results

- Metric: Measuring toxicity via the Detoxify library
- Student model: huggingface/together/meta-llama/Llama-3.2-3B-Instruct
- Teacher model(GEPA): gemini/gemini-2.0-flash
- 
### Basic performance metrics
- Base model llama without GEPA optimization on 100 samples: 99.66%
- Base model llama with GEPA optimization on 100 samples: 99.65528249740601% (GEPA did not optimize because score is too perfect)
- 
### Current limitations
- Deciding on a student model to use
    - student model performing too well -> GEPA unable to improve
    - llama-instruction-abiterated does not exist with GEPA DSPy (LiteLLM) library
 
- Deciding on Teacher model
    - Need to be smarter than student model: Gemini
    - Requires API tokens to be calling at the speed that GEPA does.
 
- GEPA unable to do anything because scoring too well.

### Resource usage
- API tokens in HuggingFace + API tokens in Gemini
- Google Colab Pro T4 GPU
---

## Next Steps

- decide on a different model or
- possible decide on a new metric of measurement for success in order to invoke GEPA
  
---

**References**
TBD
