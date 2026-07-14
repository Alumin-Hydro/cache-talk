"""Model loading and generation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


@dataclass
class GenerationResult:
    text: str
    input_ids: torch.Tensor
    output_ids: torch.Tensor
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None


class CausalLMWrapper:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_name(cls, name_or_path: str, device: str = "cpu") -> "CausalLMWrapper":
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        return cls(model, tokenizer, device=device)

    def prepare_prompt(self, prompt: str, add_generation_prompt: bool = True, use_chat_template: bool = True) -> torch.Tensor:
        if use_chat_template and hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            text = prompt
            if add_generation_prompt:
                text += "\n"
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

    def encode(self, text: str, add_special_tokens: bool = False) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids.to(self.device)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        do_sample: bool = True,
        past_key_values: Any = None,
    ) -> GenerationResult:
        """Generate tokens autoregressively with optional KV cache.

        Uses a simple loop so we can pass a pre-existing (and possibly steered) KV cache.
        """
        generated_ids = input_ids.clone()
        current_ids = input_ids
        kv_cache = past_key_values
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    current_ids,
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=False,
                )
            logits = outputs.logits[:, -1, :]
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            current_ids = next_token
            kv_cache = outputs.past_key_values

        output_ids = generated_ids[:, input_ids.shape[-1]:]
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return GenerationResult(
            text=text,
            input_ids=input_ids,
            output_ids=output_ids,
            kv_cache=kv_cache,
        )

    def __repr__(self) -> str:
        return f"CausalLMWrapper(model={type(self.model).__name__}, tokenizer={type(self.tokenizer).__name__})"
