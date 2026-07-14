"""Simple evaluation metrics."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Metrics:
    output_tokens: int
    contains_reasoning_words: bool
    reasoning_word_count: int
    final_answer_text: str


REASONING_WORDS = {
    "step", "steps", "first", "second", "third", "next", "then", "therefore", "thus", "because", "so", "finally", "answer is", "calculate", "add", "subtract", "multiply", "divide", "reason", "conclude"
}


def score_output(text: str) -> Metrics:
    tokens = text.split()
    lower = text.lower()
    reasoning_word_count = sum(1 for w in REASONING_WORDS if w in lower)
    contains_reasoning_words = reasoning_word_count >= 2

    # Extract final answer line
    final_answer = ""
    for marker in ["final answer:", "answer is", "the answer is", "therefore,", "answer:"]:
        if marker in lower:
            idx = lower.find(marker)
            final_answer = text[idx:].split("\n")[0].strip()
            break

    if not final_answer:
        # fallback: last sentence
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        final_answer = sentences[-1] if sentences else text

    return Metrics(
        output_tokens=len(tokens),
        contains_reasoning_words=contains_reasoning_words,
        reasoning_word_count=reasoning_word_count,
        final_answer_text=final_answer,
    )


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    total_tokens = sum(r["metrics"]["output_tokens"] for r in results)
    reasoning = sum(1 for r in results if r["metrics"]["contains_reasoning_words"])
    return {
        "count": len(results),
        "avg_output_tokens": round(total_tokens / len(results), 2),
        "reasoning_fraction": round(reasoning / len(results), 2),
        "avg_reasoning_word_count": round(
            sum(r["metrics"]["reasoning_word_count"] for r in results) / len(results), 2
        ),
    }
