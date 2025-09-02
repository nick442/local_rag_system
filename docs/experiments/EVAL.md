# Evaluation

This repo favors lightweight, explainable metrics for retrieval and answer quality. Choose metrics appropriate to the experiment’s goal and keep data export simple (JSONL/CSV).

## Retrieval Metrics
- Top‑k Recall: fraction of prompts where at least one expected source appears in the top‑k results.
- Source Count: number of distinct sources per query (sanity check for duplication).
- Hybrid Advantage: delta vs. pure vector/keyword for the same k.

## Answer Quality (Heuristic)
`scripts/tests/run_retrieval_tests.py` includes a simple scorer that checks overlap between response text and expected elements plus the retrieved contexts. Use it for smoke checks; rely on human spot‑checks for nuanced judgments.

## Latency/Throughput
- Response Time: wall‑clock per query (end‑to‑end).
- Tokens/sec: throughput proxy if available from the LLM wrapper.

## A/B Tests
- Compare distributions (median/mean and variance) of response time and quality.
- Report effect size when possible; fall back to descriptive stats when SciPy is unavailable.

## Reporting
- Retrieval suite emits an HTML report and machine‑readable JSONL/CSV. Persist them under a run‑specific folder in `test_results/`.
- Enable metrics for deeper timing data across components.

