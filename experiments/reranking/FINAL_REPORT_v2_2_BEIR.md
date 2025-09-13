# BEIR v2.2 Reranking Experiments — Final Report

Generated: 2025-09-13 23:00

## Abstract
We evaluate two-stage retrieval with cross-encoder reranking on BEIR FiQA and SciFact corpora. Using test-only queries aligned to BEIR qrels and k=10 evaluation, FiQA benefits from hybrid first-stage retrieval combined with MiniLM-L12 reranking, improving NDCG and recall over vector-only. SciFact shows strong gains in NDCG from the BAAI/BGE base reranker with a vector-only first stage.

## Datasets and Setup
- Datasets: FiQA (fiqa_technical), SciFact (scifact_scientific)
- Database: data/rag_vectors.db; k=10 for final evaluation; rerank-topk ∈ {50,100}
- Queries: test-only BEIR queries with IDs (test_data/beir_*_queries_test_only.json)
- Metrics: NDCG@10, Recall@10 vs BEIR qrels (evaluate_reranking.py)
- First-stage: vector, hybrid(alpha, candidate-multiplier)
- Rerankers: cross-encoder/ms-marco-MiniLM-L-12-v2, BAAI/bge-reranker-base

## Results
- FiQA baseline: ndcg=0.3766, recall=0.4591
- FiQA best: `fiqa_hybrid_a0.3_cm10_miniLM12.topk50` ndcg=0.4009, recall=0.4708 (Δ ndcg=+0.0243, Δ recall=+0.0117)
- SciFact baseline: ndcg=0.6406, recall=0.7832
- SciFact best: `scifact_rerank_bge_base.topk50` ndcg=0.8109, recall=0.8168 (Δ ndcg=+0.1703, Δ recall=+0.0336)

### FiQA — Top Runs (by NDCG@10)
| run | ndcg@10 | recall@10 | method | alpha | cm | reranker | topk |
|---|---:|---:|---|---:|---:|---|---:|
| `fiqa_hybrid_a0.3_cm10_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.3 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.3_cm3_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.3 | 3 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.3_cm5_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.3 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.5_cm10_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.5 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.5_cm3_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.5 | 3 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.5_cm5_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.5 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.7_cm10_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.7 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.7_cm3_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.7 | 3 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.7_cm5_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.7 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.9_cm10_miniLM12.topk50` | 0.4009 | 0.4708 | hybrid | 0.9 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |

### SciFact — Top Runs (by NDCG@10)
| run | ndcg@10 | recall@10 | method | alpha | cm | reranker | topk |
|---|---:|---:|---|---:|---:|---|---:|
| `scifact_rerank_bge_base.topk50` | 0.8109 | 0.8168 | vector | - | - | BAAI/bge-reranker-base | 50 |
| `scifact_rerank_bge_base.topk100` | 0.8100 | 0.8011 | vector | - | - | BAAI/bge-reranker-base | 100 |
| `scifact_rerank_msmarco_miniLM12.topk100` | 0.7991 | 0.9957 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM12.topk20` | 0.7991 | 0.9957 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM12.topk50` | 0.7973 | 1.0072 | vector | - | - | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `scifact_rerank_msmarco` | 0.7933 | 0.9957 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM6.topk100` | 0.7933 | 0.9957 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM6.topk20` | 0.7933 | 0.9957 | - | - | - | - | - |
| `scifact_hybrid_rerank_miniLM12.topk100` | 0.7888 | 0.9856 | hybrid | 0.7 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 100 |
| `scifact_rerank_bge_base.topk50.dedup` | 0.7039 | 0.8168 | vector | - | - | BAAI/bge-reranker-base | 50 |

## Discussion
- FiQA: Hybrid first-stage (alpha≈0.3–0.7, cand_mult=5) with MiniLM-L12 reranking yields the best NDCG and improves recall modestly; larger candidate sets (cm=10, topk=100) trade NDCG for recall.
- FiQA: BGE-base underperforms MiniLM-L12 on this domain/configuration; further tuning (instruction-tuned rerankers or larger BGE variants) may be required.
- SciFact: Vector-only first stage with BGE-base reranking substantially improves NDCG; recall does not match a saturated vector baseline when reranking top-k is small; increasing rerank-topk may trade latency for recall.

## Limitations and Notes
- Evaluator maps corpus-ids via filename/source stem; ensure sources preserve BEIR doc IDs.
- FTS5 warnings during keyword search are expected and safe if results are produced.
- Latency profiling is out of scope here; expected overhead rises with rerank-topk and model size.
- Deduplicated metrics are provided for reference; main comparisons use non-dedup.

## Reproducibility
- Env: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`
- Status: `RAG_SWEEP_NO_LLM=1 python main.py --db-path data/rag_vectors.db status`
- Runner (example FiQA hybrid+L12): `RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py --queries test_data/beir_fiqa_queries_test_only.json --corpus fiqa_technical --k 10 --retrieval-method hybrid --alpha 0.5 --candidate-multiplier 5 --rerank-topk 50 --reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2 --output experiments/reranking/results/fiqa_hybrid_a0.5_cm5_miniLM12.topk50.test.json`
- Evaluate: `python experiments/reranking/evaluate_reranking.py --results <results.json> --qrels <qrels.tsv> --k 10 --output <metrics.json>`
- Summarize: `python experiments/reranking/summarize_results.py --results-dir experiments/reranking/results`

## Artifacts
- Results and metrics JSONs under `experiments/reranking/results/`
- Summary: `experiments/reranking/results/summary.json`, `summary.md`