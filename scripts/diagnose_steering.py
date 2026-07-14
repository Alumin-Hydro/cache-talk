"""Quick diagnostic: compare positive vs negative reasoning prompts."""
from __future__ import annotations

import json
from pathlib import Path

from cache_talk.model import CausalLMWrapper
from cache_talk.eval import score_output


def main() -> None:
    model = CausalLMWrapper.from_name("HuggingFaceTB/SmolLM2-360M-Instruct")
    tasks = json.loads(Path("data/tasks.json").read_text())

    for task in tasks:
        q = task["question"]
        pos_prompt = f"Question: {q}\nAnswer: Let's think step by step."
        neg_prompt = f"Question: {q}\nAnswer: Direct answer follows."

        pos_ids = model.prepare_prompt(pos_prompt, add_generation_prompt=False, use_chat_template=False)
        neg_ids = model.prepare_prompt(neg_prompt, add_generation_prompt=False, use_chat_template=False)

        pos_result = model.generate(pos_ids, max_new_tokens=120)
        neg_result = model.generate(neg_ids, max_new_tokens=120)

        print(f"\n--- {task['id']} ---")
        print(f"POS tokens={len(pos_result.text.split())} reasoning={score_output(pos_result.text).contains_reasoning_words}")
        print(pos_result.text[:200].replace("\n", " "))
        print(f"NEG tokens={len(neg_result.text.split())} reasoning={score_output(neg_result.text).contains_reasoning_words}")
        print(neg_result.text[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
