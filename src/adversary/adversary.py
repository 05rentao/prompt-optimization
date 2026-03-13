import torch
import torch.nn.functional as F

class Adversary:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def policy_sample(
        model, tokenizer, messages,
        max_new_tokens=256, temperature=0.7, top_p=0.9,
        device="cuda", max_length=2048,
    ):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
            )

            prompt_len = enc["input_ids"].shape[-1]
            completion_ids = gen_ids[:, prompt_len:]  # [1, T_new]

            # logprobs as numbers (no_grad) for logging only
            out = model(input_ids=gen_ids, use_cache=False)
            logits = out.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

            start = prompt_len - 1
            end = start + completion_ids.shape[-1]
            gen_log_probs = log_probs[:, start:end, :]

            token_logprobs = torch.gather(gen_log_probs, -1, completion_ids.unsqueeze(-1)).squeeze(-1)

        completion_text = self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        return {
            "prompt": prompt,
            "prompt_len": int(prompt_len),
            "gen_ids": gen_ids,  # [1, T_full] prompt+completion
            "completion_text": completion_text,
            "completion_logprob_sum": token_logprobs.sum(dim=-1),  # [1]
        }


    def adversary_output(prompt, messages):

        res = policy_sample(
                model=self.model,
                tokenizer=self.tokenizer,
                messages=messages,
                max_new_tokens=120,
            )

        return res;