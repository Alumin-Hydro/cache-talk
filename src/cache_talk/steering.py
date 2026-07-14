"""Extract and apply KV cache steering vectors.

Targets transformers 5.x where past_key_values is a DynamicCache. We mutate the cache
in-place to avoid breaking the cache's internal bookkeeping.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from .model import CausalLMWrapper


def _get_layer_kv(past_key_values: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (K, V) tensors for a given layer."""
    if isinstance(past_key_values, DynamicCache):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    return past_key_values[layer_idx]


def _get_num_layers(past_key_values: Any) -> int:
    if isinstance(past_key_values, DynamicCache):
        return len(past_key_values.layers)
    return len(past_key_values)


def extract_steering_vector_at_answer(
    model: CausalLMWrapper,
    task: str,
    expected_answer: str = "",
    layer_idx: int | None = None,
) -> dict[str, Any]:
    """Extract steering vector at the position right after 'Answer:'.

    Positive prompt encourages detailed reasoning; negative prompt encourages a concise
    direct answer. Both share the same prefix up to and including 'Answer:'.
    """
    prefix = f"Question: {task}\nAnswer:"
    positive = f"{prefix} Let's think step by step."
    if expected_answer:
        negative = f"{prefix} The final answer is {expected_answer}."
    else:
        negative = f"{prefix} The final answer is"

    pos_ids = model.prepare_prompt(positive, add_generation_prompt=False, use_chat_template=False)
    neg_ids = model.prepare_prompt(negative, add_generation_prompt=False, use_chat_template=False)

    prefix_ids = model.tokenizer(prefix, add_special_tokens=False).input_ids
    token_idx = len(prefix_ids) - 1

    with torch.no_grad():
        pos_outputs = model.model(pos_ids, output_hidden_states=False, use_cache=True)
        neg_outputs = model.model(neg_ids, output_hidden_states=False, use_cache=True)

    pos_kv = pos_outputs.past_key_values
    neg_kv = neg_outputs.past_key_values

    if layer_idx is None:
        layer_idx = _get_num_layers(pos_kv) // 2

    pos_k, pos_v = _get_layer_kv(pos_kv, layer_idx)
    neg_k, neg_v = _get_layer_kv(neg_kv, layer_idx)

    k_diff = pos_k[:, :, token_idx, :] - neg_k[:, :, token_idx, :]
    v_diff = pos_v[:, :, token_idx, :] - neg_v[:, :, token_idx, :]

    return {
        "k": k_diff,
        "v": v_diff,
        "layer": layer_idx,
        "token": token_idx,
    }


def extract_steering_vector(
    model: CausalLMWrapper,
    positive_prompt: str,
    negative_prompt: str,
    layer_idx: int | None = None,
    token_idx: int = -1,
) -> dict[str, Any]:
    """Extract mean difference in K and V cache at token_idx for a positive vs negative prompt."""
    pos_ids = model.prepare_prompt(positive_prompt, add_generation_prompt=False, use_chat_template=False)
    neg_ids = model.prepare_prompt(negative_prompt, add_generation_prompt=False, use_chat_template=False)

    with torch.no_grad():
        pos_outputs = model.model(pos_ids, output_hidden_states=False, use_cache=True)
        neg_outputs = model.model(neg_ids, output_hidden_states=False, use_cache=True)

    pos_kv = pos_outputs.past_key_values
    neg_kv = neg_outputs.past_key_values

    if layer_idx is None:
        layer_idx = _get_num_layers(pos_kv) // 2

    pos_k, pos_v = _get_layer_kv(pos_kv, layer_idx)
    neg_k, neg_v = _get_layer_kv(neg_kv, layer_idx)

    k_diff = pos_k[:, :, token_idx, :] - neg_k[:, :, token_idx, :]
    v_diff = pos_v[:, :, token_idx, :] - neg_v[:, :, token_idx, :]

    return {
        "k": k_diff,
        "v": v_diff,
        "layer": layer_idx,
        "token": token_idx,
    }


def apply_steering_to_cache(
    past_key_values: Any,
    steering: dict[str, Any],
    c_k: float = 1.0,
    c_v: float = 1.0,
) -> Any:
    """Add scaled steering vector to K and V cache in-place at the specified layer and token position.

    Mutates the DynamicCache to preserve all internal attributes. Returns the same object.
    """
    layer_idx = steering["layer"]
    token_idx = steering["token"]
    k_vec = steering["k"] * c_k
    v_vec = steering["v"] * c_v

    if not isinstance(past_key_values, DynamicCache):
        raise TypeError("apply_steering_to_cache expects a DynamicCache instance")

    layer = past_key_values.layers[layer_idx]
    layer.keys = layer.keys.clone()
    layer.values = layer.values.clone()
    layer.keys[:, :, token_idx, :] += k_vec
    layer.values[:, :, token_idx, :] += v_vec

    return past_key_values


def make_reasoning_steering_pair(task_question: str) -> tuple[str, str]:
    """Return a positive (detailed reasoning) prompt and a negative (direct answer) prompt for a task."""
    positive = f"Question: {task_question}\nAnswer: Let's think step by step."
    negative = f"Question: {task_question}\nAnswer:"
    return positive, negative
