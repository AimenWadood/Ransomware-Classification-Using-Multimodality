from __future__ import annotations

import torch
from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

Chat = List[Dict[str, str]]

class LocalPhiModel:
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- tokenizer ---
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tok.padding_side = "left"                          # NEW
        if self.tok.pad_token is None:                          # CHG
            self.tok.pad_token = self.tok.eos_token

        # --- model ---
        dtype = (                                               # NEW
            torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float16 if self.device == "cuda" else torch.float32)
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto",
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)

        # --- workarounds ---
        self.model.config.use_cache = False
        try:
            self.model.generation_config.cache_implementation = None
        except Exception:
            pass
        try:
            self.model.generation_config.attn_implementation = "eager"
        except Exception:
            pass
        # Ensure gen config knows the special tokens                      # NEW
        self.model.generation_config.pad_token_id = self.tok.pad_token_id # NEW
        self.model.generation_config.eos_token_id = self.tok.eos_token_id # NEW

        self.model.eval()

    @staticmethod
    def _safe_seq_len(past_key_values) -> int | None:
        try:
            if past_key_values is None:
                return None
            if hasattr(past_key_values, "get_seq_length"):
                return int(past_key_values.get_seq_length())
            if isinstance(past_key_values, tuple) and past_key_values:
                k = past_key_values[0][0]
                if hasattr(k, "shape"):
                    return int(k.shape[-2])
        except Exception:
            pass
        return None

    def generate_reply(
        self,
        prompt_or_messages: Union[str, Chat],
        max_new_tokens: int = 200,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        # Build the text to feed the model
        if hasattr(self.tok, "apply_chat_template"):
            if isinstance(prompt_or_messages, str):
                messages: Chat = [
                    {"role": "system", "content": "You are a concise assistant for ML training feedback."},
                    {"role": "user", "content": prompt_or_messages},
                ]
            else:
                messages = prompt_or_messages
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt_or_messages if isinstance(prompt_or_messages, str) else "\n".join(
                f"{m.get('role','user')}: {m.get('content','')}" for m in prompt_or_messages
            )

        inputs = self.tok(text, return_tensors="pt").to(self.model.device)

        do_sample = temperature is not None and float(temperature) > 0
        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=max(0.1, float(temperature)) if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            out = self.model.generate(**inputs, use_cache=False, **gen_kwargs)  # CHG

        gen_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen_tokens, skip_special_tokens=True).strip()

    def __call__(self, prompt: Union[str, Chat], **kw) -> str:
        return self.generate_reply(prompt, **kw)
