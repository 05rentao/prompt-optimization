# this file is currently unused

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model_and_tokenizer(model_path, device="cuda", max_seq_length=1024):
    # Use bfloat16 for modern cluster GPUs (A100/H100/3090/4090)
    # Use float16 if you're on older hardware (V100/T4)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=dtype, 
        device_map=device,
        trust_remote_code=True
    )
    
    # Standard transformers doesn't "lock" the sequence length in the model 
    # like Unsloth does, so we just prep the tokenizer for it.
    tokenizer.model_max_length = max_seq_length
    
    return model, tokenizer


# You can also define a wrapper here if you use different 
# inference engines like vLLM later.

