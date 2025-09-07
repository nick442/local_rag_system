Reranking Experiments (v2.2)

Goal
- Enable two-stage retrieval with an optional CrossEncoder reranker for BEIR datasets (FiQA, SciFact).
  This branch scaffolds a minimal, non-breaking integration and runnable scripts.

What's here
- src/reranker_service.py: Optional reranker (disabled by default). Set `RAG_RERANKER_MODEL` to enable.
- RAG pipeline hook: When enabled, contexts are re-ordered before prompt building.
- Script: experiments/reranking/run_reranking_experiment.py (retrieval-only baseline vs rerank).

Prerequisites
- Conda env: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`
- Local DB: `data/rag_vectors.db` (FiQA: `fiqa_technical`, SciFact: `scifact_scientific`).

Quick start
1) Baseline (no reranking)
   RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
     --queries test_data/fiqa_queries_subset_test_100.json \
     --corpus fiqa_technical \
     --k 5 \
     --output experiments/reranking/results/fiqa_baseline.json

2) With reranking (CrossEncoder)
   export RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
   RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
     --queries test_data/fiqa_queries_subset_test_100.json \
     --corpus fiqa_technical \
     --k 5 --rerank-topk 50 \
     --output experiments/reranking/results/fiqa_rerank_msmarco_miniLM6.json

Notes
- If the reranker model fails to load, the service logs a warning and becomes a no-op.
- Default remains unchanged for existing flows and tests (no env set = identity reranker).

Next steps (for the new agent)
- Add metrics: NDCG@10/Recall@10 using existing BEIR-like tools.
- Expand run script to sweep reranker models and top-k.
- Add A/B test harness via ExperimentRunner comparing baseline vs rerank.

