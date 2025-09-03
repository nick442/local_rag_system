# Experiment 1 v2 — Chunking Optimization (Default Base) — Run Log

This run executes the `chunk_optimization` template against 20 pre‑materialized collections derived from the `default` base corpus.

## Context
- Base corpus: `default`
- Prepared collections (20): `exp_cs{128,256,512,768,1024}_co{~10%,~15%,~20%,~25%}`
- Embedding snapshot used for (re)embedding: `models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`
- LLM model: `models/gemma-3-4b-it-q4_0.gguf`
- Vector DB: `data/rag_vectors.db` (symlinked to main worktree)
- Retrieval backend: sqlite-vec (vendor or python package)

## Commands
Commands executed to prepare and run this experiment are recorded in `commands.sh` and summarized below.

1. Prepared 20 collections via `collection clone` (already done before this run)
2. Ran template experiment (small baseline queries) to validate end‑to‑end flow

## Outputs
- `results.json`: experiment results (template runner output)
- `collections_summary.csv`: chunk counts per experimental collection
- `system_info.json`: environment, versions, and paths
- `run.log`: console output (captured)

## Notes
- This run uses a small query set to complete quickly and validate correctness in this worktree. A full 52‑query run can be launched subsequently using the same folder structure.

## A/B Test Summary (Added)
- Could not parse A/B results: [Errno 2] No such file or directory: 'experiments/experiment_1_v2_run_default_base_20250903_090847/ab_results.json'
