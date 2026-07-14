"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agents import TwoAgentPipeline
from .eval import aggregate, score_output
from .model import CausalLMWrapper


def load_tasks(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def run_experiment(model: CausalLMWrapper, tasks: list[dict], condition: str, **kwargs: Any) -> list[dict]:
    pipeline = TwoAgentPipeline(model)
    records: list[dict] = []
    for task in tasks:
        task_text = task["question"]
        if condition == "baseline":
            result = pipeline.run_baseline(task_text, max_new_tokens=kwargs.get("max_new_tokens", 200))
        elif condition == "cot_prompt":
            result = pipeline.run_cot_prompt(task_text, max_new_tokens=kwargs.get("max_new_tokens", 200))
        elif condition == "text_handoff":
            result = pipeline.run_text_handoff(task_text, max_new_tokens=kwargs.get("max_new_tokens", 200))
        elif condition == "cache_steering":
            result = pipeline.run_cache_steering(
                task_text,
                expected_answer=task.get("answer", ""),
                layer_idx=kwargs.get("layer_idx"),
                token_idx=kwargs.get("token_idx", -1),
                c_k=kwargs.get("c_k", 1.0),
                c_v=kwargs.get("c_v", 1.0),
                max_new_tokens=kwargs.get("max_new_tokens", 200),
            )
        else:
            raise ValueError(f"unknown condition: {condition}")

        metrics = score_output(result.generated_text)
        records.append({
            "task_id": task.get("id"),
            "condition": condition,
            "question": task_text,
            "expected": task.get("answer"),
            "generated": result.generated_text,
            "metrics": metrics.__dict__,
            "steering_applied": result.steering_applied,
            "c_k": result.c_k,
            "c_v": result.c_v,
        })
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache Talk: multi-agent cache steering prototype")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct", help="model name or path")
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks.json"), help="task JSON file")
    parser.add_argument("--output", type=Path, default=Path("results/run.jsonl"), help="output JSONL file")
    parser.add_argument("--conditions", nargs="+", default=["baseline", "cot_prompt", "text_handoff", "cache_steering"])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--c-k", type=float, default=1.0)
    parser.add_argument("--c-v", type=float, default=1.0)
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument("--token-idx", type=int, default=-1)
    args = parser.parse_args(argv)

    if not args.tasks.exists():
        print(f"Tasks file not found: {args.tasks}", file=sys.stderr)
        return 1

    print(f"Loading model {args.model}...")
    model = CausalLMWrapper.from_name(args.model)
    tasks = load_tasks(args.tasks)
    print(f"Loaded {len(tasks)} tasks.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for condition in args.conditions:
            print(f"Running condition: {condition}")
            records = run_experiment(
                model,
                tasks,
                condition,
                max_new_tokens=args.max_new_tokens,
                c_k=args.c_k,
                c_v=args.c_v,
                layer_idx=args.layer_idx,
                token_idx=args.token_idx,
            )
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary = aggregate(records)
            print(f"  {condition}: {summary}")

    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
