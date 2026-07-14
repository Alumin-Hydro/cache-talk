"""Grid search over steering strengths and injection positions."""
from __future__ import annotations

import json
from pathlib import Path

from cache_talk.agents import TwoAgentPipeline
from cache_talk.eval import aggregate, score_output
from cache_talk.model import CausalLMWrapper


def grid_search(
    model: CausalLMWrapper,
    tasks: list[dict],
    c_values: list[float] = [0.05, 0.1, 0.2, 0.5, 1.0],
    layer_indices: list[int | None] = [None, 5, 10, 15],
) -> list[dict]:
    pipeline = TwoAgentPipeline(model)
    records: list[dict] = []

    for c in c_values:
        for layer_idx in layer_indices:
            for task in tasks:
                result = pipeline.run_cache_steering(
                    task["question"],
                    expected_answer=task.get("answer", ""),
                    layer_idx=layer_idx,
                    c_k=c,
                    c_v=c,
                    max_new_tokens=120,
                )
                metrics = score_output(result.generated_text)
                records.append({
                    "task_id": task.get("id"),
                    "c": c,
                    "layer": layer_idx if layer_idx is not None else "mid",
                    "question": task["question"],
                    "generated": result.generated_text,
                    "metrics": metrics.__dict__,
                })

    return records


def main() -> None:
    model = CausalLMWrapper.from_name("HuggingFaceTB/SmolLM2-360M-Instruct")
    tasks = json.loads(Path("data/tasks.json").read_text())
    records = grid_search(model, tasks[:3])  # use first 3 tasks for speed

    output = Path("results/grid_search.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary by (c, layer)
    from collections import defaultdict
    grouped: dict[tuple, list] = defaultdict(list)
    for r in records:
        grouped[(r["c"], r["layer"])].append(r)

    print("\n=== Grid search summary ===")
    items = []
    for (c, layer), group in grouped.items():
        layer_key = -1 if layer == "mid" else layer
        avg_tokens = sum(r["metrics"]["output_tokens"] for r in group) / len(group)
        avg_reasoning = sum(r["metrics"]["reasoning_word_count"] for r in group) / len(group)
        items.append((c, layer_key, layer, avg_tokens, avg_reasoning))
    items.sort()
    for c, _, layer, avg_tokens, avg_reasoning in items:
        print(f"c={c}, layer={layer}: avg_tokens={avg_tokens:.1f}, avg_reasoning={avg_reasoning:.1f}")


if __name__ == "__main__":
    main()
