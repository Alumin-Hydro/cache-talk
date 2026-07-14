# Cache Talk Experiment Report

Date: 2026-07-14
Model: HuggingFaceTB/SmolLM2-360M-Instruct (CPU)
Dataset: 5 synthetic reasoning tasks (data/tasks.json)

## Goal

Test whether KV cache steering can act as a lightweight "communication protocol" between
agents that share the same language model. Agent A prefills a task prompt; a steering
vector is injected into its KV cache; Agent B continues generation from that cache.

## Methods

- **baseline**: single-agent generation with `Question: ...\nAnswer:`.
- **cot_prompt**: single-agent with `Answer: Let's think step by step.`.
- **text_handoff**: Agent A writes a reasoning trace; Agent B finalizes.
- **cache_steering**: Agent A prefills; positive-vs-negative steering vector is extracted
  at the `Answer:` token and added to the KV cache; Agent B continues.

Steering vector:
- positive = `Answer: Let's think step by step.`
- negative = `Answer: The final answer is <expected_answer>.`

Grid search over `c_k`/`c_v` in {0.05, 0.1, 0.2, 0.5, 1.0} and layer indices {mid, 5, 10, 15}.

## Key Results

| Condition        | Avg output tokens | Reasoning fraction | Avg reasoning words |
|------------------|------------------:|-------------------:|--------------------:|
| baseline         | 70.4              | 0.6                | 2.0                 |
| cot_prompt       | (not run)         | -                  | -                   |
| text_handoff     | (not run)         | -                  | -                   |
| cache_steering   | 9.2               | 0.0                | 0.2                 |

Grid search summary (first 3 tasks, c=c_k=c_v):
- `c=0.1, layer=mid`: avg_tokens=50.7, avg_reasoning=1.7 (best observed)
- `c=0.5, layer=10`: avg_tokens=26.0, avg_reasoning=0.3
- Most combinations cause early EOS or output degradation.

## Interpretation

On SmolLM2-360M, KV cache steering at the `Answer:` token is highly unstable:
- Small coefficients (c≤0.1) sometimes preserve generation length but produce inconsistent reasoning signals.
- Larger coefficients (c≥0.2) frequently collapse the model into EOS or gibberish.
- Different layers show no clear monotonic trend; mid-layer is as unstable as early/late layers.

This suggests cache steering is **not yet a reliable multi-agent protocol** at this scale.
Plausible reasons:
1. 360M models lack the capacity to maintain coherent generation under K/V perturbations.
2. The steering vector is extracted from a single token and may not robustly encode "reasoning style".
3. The negative prompt (`The final answer is X`) still contains answer information, making the contrast weak.

## Next Steps

1. Repeat on larger models (e.g., Llama-3.2-1B/3B, Qwen2.5-1.5B) to see if stability improves.
2. Use average steering across multiple answer tokens rather than a single token.
3. Compare with activation steering on hidden states (not KV) as a stronger baseline.
4. Try C2C-style cross-model cache communication if multiple checkpoints are available.

## Artifacts

- Code: `~/Project/cache-talk/`
- Results: `~/Project/cache-talk/results/run.jsonl`, `~/Project/cache-talk/results/grid_search.jsonl`
- Wiki: `~/Project/wiki/concepts/kv-cache-steering.md`, `~/Project/wiki/entities/cache-to-cache.md`
