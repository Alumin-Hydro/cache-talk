"""Agent pipeline: Agent A prefill, optional steering injection, Agent B continuation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .model import CausalLMWrapper
from .steering import apply_steering_to_cache, extract_steering_vector_at_answer, make_reasoning_steering_pair


@dataclass
class AgentTurnResult:
    agent_name: str
    prompt: str
    generated_text: str
    kv_cache: Any
    steering_applied: bool
    c_k: float | None
    c_v: float | None


class TwoAgentPipeline:
    def __init__(self, model: CausalLMWrapper) -> None:
        self.model = model

    def _format(self, task: str, suffix: str = "") -> str:
        return f"Question: {task}\n{suffix}"

    def run_baseline(self, task: str, max_new_tokens: int = 200) -> AgentTurnResult:
        """Single agent with no steering."""
        prompt = self._format(task, "Answer:")
        input_ids = self.model.prepare_prompt(prompt, use_chat_template=False)
        result = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        return AgentTurnResult(
            agent_name="baseline",
            prompt=prompt,
            generated_text=result.text,
            kv_cache=result.kv_cache,
            steering_applied=False,
            c_k=None,
            c_v=None,
        )

    def run_cot_prompt(self, task: str, max_new_tokens: int = 200) -> AgentTurnResult:
        """Single agent with text-only CoT prompt."""
        prompt = self._format(task, "Answer: Let's think step by step.")
        input_ids = self.model.prepare_prompt(prompt, use_chat_template=False)
        result = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        return AgentTurnResult(
            agent_name="cot_prompt",
            prompt=prompt,
            generated_text=result.text,
            kv_cache=result.kv_cache,
            steering_applied=False,
            c_k=None,
            c_v=None,
        )

    def run_text_handoff(
        self,
        task: str,
        agent_a_prompt: str = "You are a helpful assistant. Analyze the following question and produce a brief reasoning trace.",
        max_new_tokens: int = 200,
    ) -> AgentTurnResult:
        """Agent A generates text reasoning, Agent B sees task + reasoning."""
        a_prompt = f"{agent_a_prompt}\n\nQuestion: {task}\nReasoning:"
        a_input = self.model.prepare_prompt(a_prompt, use_chat_template=False)
        a_result = self.model.generate(a_input, max_new_tokens=max_new_tokens)

        handoff = f"Question: {task}\nReasoning: {a_result.text.strip()}\nFinal answer:"
        b_input = self.model.prepare_prompt(handoff, use_chat_template=False)
        b_result = self.model.generate(b_input, max_new_tokens=max_new_tokens)

        return AgentTurnResult(
            agent_name="text_handoff",
            prompt=handoff,
            generated_text=b_result.text,
            kv_cache=b_result.kv_cache,
            steering_applied=False,
            c_k=None,
            c_v=None,
        )

    def run_cache_steering(
        self,
        task: str,
        expected_answer: str = "",
        layer_idx: int | None = None,
        token_idx: int = -1,
        c_k: float = 1.0,
        c_v: float = 1.0,
        max_new_tokens: int = 200,
    ) -> AgentTurnResult:
        """Agent A prefills the task; a steering vector is injected; Agent B continues generation."""
        prompt = f"Question: {task}\nAnswer:"
        a_input = self.model.prepare_prompt(prompt, add_generation_prompt=False, use_chat_template=False)
        with torch.no_grad():
            a_outputs = self.model.model(a_input, output_hidden_states=False, use_cache=True)

        steering = extract_steering_vector_at_answer(
            self.model, task, expected_answer=expected_answer, layer_idx=layer_idx
        )

        steered_kv = apply_steering_to_cache(a_outputs.past_key_values, steering, c_k=c_k, c_v=c_v)

        b_input = a_input[:, -1:]
        result = self.model.generate(b_input, max_new_tokens=max_new_tokens, past_key_values=steered_kv)

        return AgentTurnResult(
            agent_name="cache_steering",
            prompt=prompt,
            generated_text=result.text,
            kv_cache=result.kv_cache,
            steering_applied=True,
            c_k=c_k,
            c_v=c_v,
        )
