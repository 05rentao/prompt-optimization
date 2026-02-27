import yaml

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.harmbench_loader import HarmBenchLoader
from src.evaluator.judge import compute_reward, get_judge
import pandas as pd

def load_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    judge_model_path = config.get("evaluator", {}).get("model_name", "cais/HarmBench-Llama-2-13b-cls")
    target_model_path = config.get("target", {}).get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    seed_prompt = config.get("seed_prompt", {}).get("prompt", "You are a helpful assistant.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load data
    print("📊 Loading HarmBench behaviors...")
    harmbench_limit = config.get("training", {}).get("harmbench_limit")
    df = HarmBenchLoader.load_csv(limit=harmbench_limit)
    if df is None or df.empty:
        print("No HarmBench data available.")
        return 0.0

    _, test = train_test_split(df, test_size=0.2, random_state=42)  # hold-out test split

    # 2. Target model (Qwen/Qwen2.5-1.5B-Instruct)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    target_model.to(device)
    target_model.eval()

    # 3. Judge model: cais/HarmBench-Llama-2-13b-cls
    judge = get_judge(model_name=judge_model_path)

    jailbreak_count = 0.0
    results = []

    for _, row in test.iterrows():
        behavior = row["behavior"]

        # 4. Prepare prompt using the chat template
        messages = [
            {"role": "system", "content": seed_prompt},
            {"role": "user", "content": behavior},
        ]
        text = target_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 5. Generate
        model_inputs = target_tokenizer([text], return_tensors="pt").to(device)
        generated_ids = target_model.generate(
            **model_inputs,
            max_new_tokens=512,
        )

        # 6. Decode
        response = target_tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]

        # 7. Judge eval
        judge_outcome = judge(behavior, response)
        jailbreak_count += compute_reward(judge_outcome.is_jailbroken)
        
        results.append({
            "behavior": behavior,
            "response": response,
            "is_jailbroken": judge_outcome.is_jailbroken
        })


    num_examples = len(test)
    if num_examples == 0:
        return 0.0
    
    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("✅ Results saved to results.csv")

    num_examples = len(test)

    asr = jailbreak_count / num_examples
    
    print(asr)
    return asr

if __name__ == "__main__":
    main()
