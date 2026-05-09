# experiment description:

Phase 2: Co-Evolution Experiments (Days 4–6)
Run R14: Full Co-Evolution with RLOO
Previous co-evolution experiments used REINFORCE, which we now know is inferior. This run uses the R11 configuration (RLOO + length penalty) in the co-evolution loop to answer the second research question about co-evolutionary robustness.
•	Configuration: RLOO + length penalty adversary, GEPA defender
•	Target: 60 adversary iterations alternating with GEPA updates
•	Track ASR at each stage boundary
•	Look for oscillation patterns or arms-race dynamics
Document Convergence Behavior
•	Log ASR, refusal rate, and aggregate score at each stage
•	Save GEPA-evolved system prompts at each iteration
•	Note any instabilities or mode collapse during alternation



# adversarial setting 

adversary=Qwen2.5-7B-bnb-4bit | target=Llama-3.1-8B | judge=HarmBench-Mistral-7b | policy=RLOO | iters=120 | batch=4 | length_penalty=0.2 | seed=42	