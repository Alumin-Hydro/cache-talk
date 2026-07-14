# Cache Talk

Prototype: can KV cache steering act as a lightweight communication protocol between agents that share the same language model?

## Idea

- Agent A processes a task prompt.
- We extract or inject a KV cache steering vector that biases the model toward detailed reasoning.
- Agent B continues from the same KV cache and generates the final answer.
- Compare against baselines: no steering, text-only CoT prompt, text handoff from Agent A to Agent B.

## Usage

```bash
uv run python -m cache_talk --model HuggingFaceTB/SmolLM2-360M-Instruct \
                            --tasks data/tasks.json \
                            --output results/run_1.jsonl
```

## Project structure

- `src/cache_talk/model.py`: model loading and generation helpers
- `src/cache_talk/steering.py`: extract and apply KV steering vectors
- `src/cache_talk/agents.py`: Agent A / Agent B pipeline
- `src/cache_talk/eval.py`: metrics and scoring
- `src/cache_talk/cli.py`: command-line entry point
- `data/tasks.json`: simple task dataset
- `results/`: experiment outputs
