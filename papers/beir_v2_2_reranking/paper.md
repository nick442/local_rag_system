# Two-Stage Retrieval with Cross-Encoder Reranking on BEIR FiQA and SciFact

## Abstract
We study two-stage retrieval for Retrieval-Augmented Generation (RAG) on BEIR datasets, combining first-stage vector/keyword/hybrid retrieval with cross-encoder reranking. Using test-only BEIR queries with ground-truth qrels and k=10 evaluation, we compare MiniLM cross-encoders and the BAAI/BGE reranker across FiQA (financial QA) and SciFact (scientific claim verification). On FiQA, hybrid first-stage retrieval plus MiniLM‑L12 reranking improves NDCG@10 from 0.3766 to 0.4009 and Recall@10 from 0.4591 to 0.4708. On SciFact, vector‑only first stage with BGE reranking yields a large NDCG@10 improvement from 0.6406 to 0.8109, with Recall@10 improving from 0.7832 to 0.8168. We discuss the interplay between first-stage candidate quality, rerank-topk, and reranker choice, and provide a fully reproducible pipeline, evaluation, and report.

## 1. Introduction
Two-stage retrieval pipelines improve precision in IR and RAG, where high-quality contexts enhance grounded responses. We evaluate this design on BEIR FiQA and SciFact, using an SQLite+sqlite‑vec vector index, FTS5 keyword search, optional hybrid fusion, and cross-encoder reranking.

## 2. Related Work
- BEIR benchmark
- Cross-encoder reranking (MS MARCO MiniLM)
- BAAI/BGE reranker
- Hybrid retrieval and fusion (max-norm, z-score, RRF)

## 3. Datasets
- FiQA (fiqa_technical): 648 test queries
- SciFact (scifact_scientific): 300 test queries
Test-only queries include explicit IDs; evaluation aligns retrieved docs via source filename stem.

## 4. Methods
- First stage: vector, keyword, or hybrid (alpha∈{0.3,0.5,0.7,0.9}, candidate-multiplier∈{3,5,10})
- Second stage: MiniLM CrossEncoders (L‑6, L‑12), BGE (BAAI/bge-reranker-base); rerank-topk∈{20,50,100}
- Evaluation: NDCG@10, Recall@10 vs BEIR qrels; k=10 final contexts

## 5. Setup
- DB: `data/rag_vectors.db` (FiQA, SciFact)
- Env: `conda activate rag_env`; `RAG_SWEEP_NO_LLM=1`
- Scripts: `experiments/reranking/run_reranking_experiment.py`, `evaluate_reranking.py`

## 6. Results
### FiQA (k=10)
- Baseline vector: ndcg=0.3766, recall=0.4591
- Hybrid+MiniLM‑L12 (topk50, α≈0.3–0.7, cm=5): ndcg=0.4009, recall=0.4708
- Hybrid+MiniLM‑L12 (topk100, α=0.5, cm=10): ndcg=0.3935, recall=0.4802
- BGE (vector, topk100): ndcg=0.3456, recall=0.4193
- BGE (hybrid, a=0.3, cm=5, topk50): ndcg=0.3592, recall=0.4351

### SciFact (k=10)
- Baseline vector: ndcg=0.6406, recall=0.7832
- BGE (vector, topk50): ndcg=0.8109, recall=0.8168
- BGE (vector, topk100): ndcg=0.8100, recall=0.8011

## 7. Analysis
- FiQA: Hybrid+MiniLM‑L12 best overall; cm/topk trades recall vs NDCG.
- SciFact: BGE strongly improves NDCG; topk50 preferable to topk100 for recall.

## 8. Limitations
- Only FiQA and SciFact; no latency profiling; evaluator depends on filename-based ID mapping; limited hyperparameter grid.

## 9. Reproducibility
- See `experiments/reranking/results/summary.md` and `experiments/reranking/FINAL_REPORT_v2_2_BEIR.md`.
- Example FiQA hybrid+L12: `RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py --queries test_data/beir_fiqa_queries_test_only.json --corpus fiqa_technical --k 10 --retrieval-method hybrid --alpha 0.5 --candidate-multiplier 5 --rerank-topk 50 --reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2 --output experiments/reranking/results/fiqa_hybrid_a0.5_cm5_miniLM12.topk50.test.json`

## 10. Conclusions
Two-stage retrieval with reranking improves retrieval quality; optimal configuration is dataset-dependent. FiQA favors hybrid+MiniLM‑L12; SciFact benefits most from BGE with vector-only first stage.

## References (stub)
[1] BEIR Benchmark (arXiv:2104.08663)  
[2] Sentence-BERT (EMNLP 2019)  
[3] BGE Reranker (arXiv)
