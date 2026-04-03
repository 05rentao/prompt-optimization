"""Adversary runtime with LoRA-capable policy sampling (standard transformers+peft, no unsloth)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as f
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

from .config import UnslothAdversaryConfig
from .interfaces import LoRABridge


class UnslothAdversaryRuntime(LoRABridge):
    """Wraps a PEFT LoRA model and exposes sampling/save helpers."""
    _sample_lock = threading.Lock()

    def __init__(self, cfg: UnslothAdversaryConfig) -> None:
        """Load base model and attach LoRA adapters."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if cfg.load_in_4bit else None

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        base_model.config.use_cache = False

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __getattr__(self, name: str) -> Any:
        """Delegate nn.Module attributes to underlying PEFT model."""
        return getattr(self.model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate direct forward calls to the wrapped model."""
        return self.model(*args, **kwargs)

    def sample_policy(
        self,
        messages: list[dict[str, str]],
        device: str,
        max_new_tokens: int = 256,  # Run entrypoints pass YAML-derived limits; these are library fallbacks.
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_length: int = 2048,
    ) -> dict[str, Any]:
        """Sample completion text and token log-prob aggregates."""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with self._sample_lock:
            self.model.eval()
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                )
                prompt_len = enc["input_ids"].shape[-1]
                completion_ids = gen_ids[:, prompt_len:]

                out = self.model(input_ids=gen_ids, use_cache=False)
                logits = out.logits
                log_probs = f.log_softmax(logits[:, :-1, :], dim=-1)

                start = prompt_len - 1
                end = start + completion_ids.shape[-1]
                gen_log_probs = log_probs[:, start:end, :]
                token_logprobs = torch.gather(gen_log_probs, -1, completion_ids.unsqueeze(-1)).squeeze(-1)

        completion_text = self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        return {
            "prompt": prompt,
            "prompt_len": int(prompt_len),
            "gen_ids": gen_ids,
            "completion_text": completion_text,
            "completion_logprob_sum": token_logprobs.sum(dim=-1),
        }

    def save_adapters(self, save_dir: Path) -> None:
        """Persist LoRA adapters and tokenizer to disk."""
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
