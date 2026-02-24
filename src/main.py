import json
from models import load_model_and_tokenizer
from evaluator import JailbreakEvaluator
from steering_methods.activation_add import ActivationAddition
import pandas as pd

def main():

    target_model, target_tok = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    
    judge_model, judge_tok = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")  
    
    dataset = pd.read_csv('data/jailbreak_10.csv')
    # index,goal,target,behavior,category,source
    evaluator = JailbreakEvaluator(target_model, judge_model, target_tok, judge_tok)

    # 2. Run Baseline (No Steering)
    print("Running Baseline...")
    baseline_results = evaluator.run_eval(dataset)
    
    # 2. Run Steering Experiment
    print("Running Steering Method A...")
    # Initialize the steerer (it starts completely dormant)
    steerer = ActivationAddition(target_model, target_tok)
    
    # Load your extracted vector. 
    # (Assuming it's saved as a standard PyTorch tensor)
    refusal_vector = torch.load("vectors/refusal_vector_layer15.pt")

    # Apply the hook! (Targeting layer 15 with a coefficient of 2.0)
    steerer.apply(layer_idx=15, steering_vector=refusal_vector, coefficient=2.0)
    
    # The evaluator runs exactly as before, but the target_model is now hooked
    steered_results = evaluator.run_eval(dataset)
    
    # Clean up (remove the hook so the model returns to normal)
    steerer.remove()

    # 3. Compare Results
    print(f"Baseline ASR: {baseline_results['success_rate']}")
    print(f"Steered ASR: {steered_results['success_rate']}")

if __name__ == "__main__":
    main()