import json
import os
import re

import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.harmbench_loader import HarmBenchLoader
from src.evaluator.judge import compute_reward, get_judge
from src.steering_methods.activation_add import ActivationAddition, extract_steering_vectors


CAA_DATA_PATH = "data/refusal_CAA_training.json"
VECTOR_DIR = "vectors"
OUTPUT_CSV_PATH = "outputs/steered_results.csv"
OUTPUT_SUMMARY_PATH = "outputs/steered_summary.json"
TARGET_LAYERS = [15]
STEERING_COEFFICIENT = 1.0
MAX_NEW_TOKENS = 512


def load_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def _extract_question_stem(question: str) -> str:
    if "\n\nChoices:\n" in question:
        return question.split("\n\nChoices:\n", 1)[0].strip()
    return question.strip()


def _extract_choices(question: str):
    choices_text = question.split("\n\nChoices:\n", 1)[1] if "\n\nChoices:\n" in question else question
    matches = re.findall(r"\(([A-Z])\)\s*(.*?)(?=\n\([A-Z]\)\s*|$)", choices_text, flags=re.DOTALL)
    return {label: text.strip() for label, text in matches}


def _normalize_label(label: str) -> str:
    label = (label or "").strip().upper()
    if label.startswith("(") and label.endswith(")"):
        label = label[1:-1]
    return label


def build_caa_pairs(caa_path: str):
    with open(caa_path, "r") as f:
        data = json.load(f)

    pos_data = []
    neg_data = []
    skipped = 0

    for row in data:
        question = row.get("question", "")
        pos_label = _normalize_label(row.get("answer_matching_behavior"))
        neg_label = _normalize_label(row.get("answer_not_matching_behavior"))
        choices = _extract_choices(question)
        instruction = _extract_question_stem(question)

        pos_response = choices.get(pos_label)
        neg_response = choices.get(neg_label)
        if not instruction or not pos_response or not neg_response:
            skipped += 1
            continue

        pos_data.append({"instruction": instruction, "response": pos_response})
        neg_data.append({"instruction": instruction, "response": neg_response})

    if not pos_data or not neg_data:
        raise ValueError("Failed to build non-empty pos/neg CAA datasets from refusal_CAA_training.json.")

    print(f"Built CAA pairs: pos={len(pos_data)}, neg={len(neg_data)}, skipped={skipped}")
    return pos_data, neg_data


def main():
    config = load_config()
    judge_model_path = config.get("evaluator", {}).get("model_name", "cais/HarmBench-Llama-2-13b-cls")
    target_model_path = config.get("target", {}).get("hf_model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    seed_prompt = config.get("seed_prompt", {}).get("prompt", "You are a helpful assistant.")
    harmbench_limit = config.get("training", {}).get("harmbench_limit")

    os.makedirs(VECTOR_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("Loading HarmBench behaviors...")
    df = HarmBenchLoader.load_csv(limit=harmbench_limit)
    if df is None or df.empty:
        print("No HarmBench data available.")
        return 0.0

    if len(df) <= 20:
        test = df.copy()
    else:
        _, test = train_test_split(df, test_size=20, random_state=42)

    print(f"Loading target model: {target_model_path}")
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        target_model.to("cuda")
    target_model.eval()

    print(f"Loading judge model endpoint for: {judge_model_path}")
    # judge = get_judge(model_name=judge_model_path)

    print("Building CAA positive/negative pairs...")
    pos_data, neg_data = build_caa_pairs(CAA_DATA_PATH)

    print(f"Extracting steering vectors for layers: {TARGET_LAYERS}")
    vectors = extract_steering_vectors(
        model=target_model,
        tokenizer=target_tokenizer,
        pos_data=pos_data,
        neg_data=neg_data,
        target_layers=TARGET_LAYERS,
    )

    for layer_idx, vector in vectors.items():
        save_path = os.path.join(VECTOR_DIR, f"refusal_vector_layer{layer_idx}.pt")
        torch.save(vector, save_path)
        print(f"Saved vector: {save_path}")

    selected_layer = TARGET_LAYERS[0]
    steerer = ActivationAddition(target_model, target_tokenizer)
    steerer.apply(
        layer_idx=selected_layer,
        steering_vector=vectors[selected_layer],
        coefficient=STEERING_COEFFICIENT,
    )

    jailbreak_count = 0.0
    results = []
    num_examples = len(test)

    try:
        for _, row in test.iterrows():
            behavior = row["behavior"]
            messages = [
                {"role": "system", "content": seed_prompt},
                {"role": "user", "content": behavior},
            ]
            text = target_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_inputs = target_tokenizer([text], return_tensors="pt").to(next(target_model.parameters()).device)
            generated_ids = target_model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            response = target_tokenizer.batch_decode(
                generated_ids[:, model_inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )[0]

            judge_outcome = judge(behavior, response)
            jailbreak_count += compute_reward(judge_outcome.is_jailbroken)

            results.append(
                {
                    "behavior": behavior,
                    "response": response,
                    "is_jailbroken": judge_outcome.is_jailbroken,
                }
            )
    finally:
        steerer.remove()

    if num_examples == 0:
        return 0.0

    pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Results saved to {OUTPUT_CSV_PATH}")

    asr = jailbreak_count / num_examples
    summary = {
        "target_model": target_model_path,
        "judge_model": judge_model_path,
        "dataset_size": num_examples,
        "target_layers": TARGET_LAYERS,
        "selected_layer": selected_layer,
        "coefficient": STEERING_COEFFICIENT,
        "asr": asr,
        "results_csv": OUTPUT_CSV_PATH,
    }
    with open(OUTPUT_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {OUTPUT_SUMMARY_PATH}")
    print(asr)
    return asr


if __name__ == "__main__":
    main()
