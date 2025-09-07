Handoff: v2.2 Reranking Experiments (BEIR)

Objective
- Enable two-stage retrieval with an optional CrossEncoder reranker on BEIR datasets (FiQA, SciFact) with minimal, non-breaking integration.

Prepared
- src/reranker_service.py: Optional reranker (disabled by default). Enable via RAG_RERANKER_MODEL or the runner flag.
- src/rag_pipeline.py: Hook to rerank contexts before prompt building when enabled.
- experiments/reranking/run_reranking_experiment.py: Retrieval-only baseline vs reranked runs.
- experiments/reranking/README.md: Quick start, examples, and next steps.

Environment
- Activate: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`
- DB: `data/rag_vectors.db` with collections: `fiqa_technical`, `scifact_scientific`.

Quick Commands
1) Baseline (vector-only):
```
RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
  --queries test_data/fiqa_queries_subset_test_100.json \
  --corpus fiqa_technical --k 5 \
  --output experiments/reranking/results/fiqa_baseline.json
```

2) Reranked (CrossEncoder):
```
export RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
  --queries test_data/fiqa_queries_subset_test_100.json \
  --corpus fiqa_technical --k 5 --rerank-topk 50 \
  --output experiments/reranking/results/fiqa_rerank_msmarco_miniLM6.json
```

Where to Extend
- Add metrics (NDCG@10/Recall@10) using existing evaluation utilities.
- Add sweeps over rerankers and top-k; integrate with ExperimentRunner for A/B.
- Expand to SciFact and full BEIR sets; add analysis/visualization scripts.

Rollback
- No default changes. Reranker is noop unless RAG_RERANKER_MODEL is set or passed.

