"""Local Transformers-based target runtime implementation."""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import LocalHFConfig
from .env import resolve_hf_token
from .interfaces import GenerationRequest, TargetRuntime


class LocalHFChatRuntime(TargetRuntime):
    """Runs local HF chat generation with optional 4-bit loading."""

    def __init__(self, cfg: LocalHFConfig) -> None:
        """Load tokenizer/model and freeze parameters for inference."""
        token = resolve_hf_token()

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        bnb_cfg = None
        if cfg.use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            token=token,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def generate(self, request: GenerationRequest, device: str) -> str:
        """Generate one response from a chat-style request payload."""
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device)

        do_sample = request.temperature > 0.0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=do_sample,
                temperature=request.temperature if do_sample else None,
                top_p=request.top_p if do_sample else None,
                use_cache=True,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

