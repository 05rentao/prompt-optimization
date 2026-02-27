"""
Target model callable using a small HuggingFace model (e.g. Qwen) in-process.
No Ollama or API needed; uses transformers (optional 4-bit to save VRAM).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 256


def _parse_system_user(full_input: str):
    """Parse 'System: ...\\nUser: ...' into (system, user)."""
    if "\nUser:" in full_input:
        parts = full_input.split("\nUser:", 1)
        system = parts[0].replace("System:", "").strip() if parts[0] else ""
        user = parts[1].strip() if len(parts) > 1 else full_input
        return system, user
    return "", full_input


def make_hf_target(
    model_name: str = DEFAULT_HF_MODEL,
    load_in_4bit: bool = True,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    Load a small chat model (e.g. Qwen2.5-0.5B-Instruct) and return a callable.
    full_input should be "System: ...\\nUser: ...". Uses the model's chat template.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if kwargs.get("device_map") is None and torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    def target_model(full_input: str) -> str:
        system, user = _parse_system_user(full_input)
        messages = [{"role": "system", "content": system or "You are a helpful assistant."}, {"role": "user", "content": user}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            return reply
        except Exception as e:
            return f"[HF target error: {e}]"

    return target_model
