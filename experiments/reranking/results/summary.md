## FIQA Results

| run | ndcg@10 | recall@10 | n | method | alpha | cand_mult | reranker | rerank_topk |
|---|---:|---:|---:|---|---:|---:|---|---:|
| `fiqa_hybrid_a0.3_cm10_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.3 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.3_cm3_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.3 | 3 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.3_cm5_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.3 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.5_cm10_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.5 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.5_cm3_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.5 | 3 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.5_cm5_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.5 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.7_cm10_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.7 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.7_cm3_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.7 | 3 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.7_cm5_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.7 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.9_cm10_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.9 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.9_cm3_miniLM12.topk50` | 0.4009 | 0.4708 | 648 | hybrid | 0.9 | 3 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.5_cm10_miniLM12.topk100` | 0.3935 | 0.4802 | 648 | hybrid | 0.5 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 100 |
| `fiqa_hybrid_rerank_miniLM12.topk100` | 0.3935 | 0.4802 | 648 | hybrid | 0.7 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 100 |
| `fiqa_rerank_msmarco_miniLM12.topk100` | 0.3926 | 0.4591 | 648 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM12.topk20` | 0.3926 | 0.4591 | 648 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM12.topk50` | 0.3926 | 0.4591 | 648 | - | - | - | - | - |
| `fiqa_hybrid_a0.3_cm5_miniLM12.topk50.dedup` | 0.3909 | 0.4671 | 648 | hybrid | 0.3 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_hybrid_a0.5_cm5_miniLM12.topk50.dedup` | 0.3909 | 0.4671 | 648 | hybrid | 0.5 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `fiqa_rerank_msmarco` | 0.3886 | 0.4591 | 648 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM6.topk100` | 0.3886 | 0.4591 | 648 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM6.topk20` | 0.3886 | 0.4591 | 648 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM12.topk100.dedup` | 0.3848 | 0.4428 | 648 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM12.topk20.dedup` | 0.3848 | 0.4428 | 648 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM12.topk50.dedup` | 0.3848 | 0.4428 | 648 | - | - | - | - | - |
| `fiqa_hybrid_a0.5_cm10_miniLM12.topk100.dedup` | 0.3828 | 0.4580 | 648 | hybrid | 0.5 | 10 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 100 |
| `fiqa_hybrid_rerank_miniLM12.topk100.dedup` | 0.3828 | 0.4580 | 648 | hybrid | 0.7 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 100 |
| `fiqa_rerank_msmarco_miniLM6.dedup` | 0.3811 | 0.4428 | 648 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM6.topk100.dedup` | 0.3811 | 0.4428 | 648 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM6.topk20.dedup` | 0.3811 | 0.4428 | 648 | - | - | - | - | - |
| `fiqa_baseline` | 0.3766 | 0.4591 | 648 | vector | - | - | - | - |
| `fiqa_baseline.dedup` | 0.3692 | 0.4428 | 648 | vector | - | - | - | - |
| `fiqa_hybrid_a0.3_cm5_bge_base.topk50` | 0.3592 | 0.4351 | 648 | hybrid | 0.3 | 5 | BAAI/bge-reranker-base | 50 |
| `fiqa_hybrid_a0.5_cm10_bge_base.topk100` | 0.3456 | 0.4193 | 648 | hybrid | 0.5 | 10 | BAAI/bge-reranker-base | 100 |
| `fiqa_hybrid_rerank_bge_base.topk100` | 0.3456 | 0.4193 | 648 | hybrid | 0.7 | 5 | BAAI/bge-reranker-base | 100 |
| `fiqa_rerank_bge_base.topk100` | 0.3456 | 0.4193 | 648 | vector | - | - | BAAI/bge-reranker-base | 100 |
| `fiqa_hybrid_rerank_bge_base.topk100.dedup` | 0.3355 | 0.4145 | 648 | hybrid | 0.7 | 5 | BAAI/bge-reranker-base | 100 |
| `fiqa_rerank_bge_base.topk100.dedup` | 0.3355 | 0.4145 | 648 | vector | - | - | BAAI/bge-reranker-base | 100 |
| `fiqa_baseline.small200` | 0.0000 | 0.0000 | 200 | - | - | - | - | - |
| `fiqa_rerank_msmarco_miniLM6.small200` | 0.0000 | 0.0000 | 200 | - | - | - | - | - |

## SCIFACT Results

| run | ndcg@10 | recall@10 | n | method | alpha | cand_mult | reranker | rerank_topk |
|---|---:|---:|---:|---|---:|---:|---|---:|
| `scifact_rerank_bge_base.topk50` | 0.8109 | 0.8168 | 300 | vector | - | - | BAAI/bge-reranker-base | 50 |
| `scifact_rerank_bge_base.topk100` | 0.8100 | 0.8011 | 300 | vector | - | - | BAAI/bge-reranker-base | 100 |
| `scifact_rerank_msmarco_miniLM12.topk100` | 0.7991 | 0.9957 | 300 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM12.topk20` | 0.7991 | 0.9957 | 300 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM12.topk50` | 0.7973 | 1.0072 | 300 | vector | - | - | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `scifact_rerank_msmarco` | 0.7933 | 0.9957 | 300 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM6.topk100` | 0.7933 | 0.9957 | 300 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM6.topk20` | 0.7933 | 0.9957 | 300 | - | - | - | - | - |
| `scifact_hybrid_rerank_miniLM12.topk100` | 0.7888 | 0.9856 | 300 | hybrid | 0.7 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 100 |
| `scifact_rerank_bge_base.topk50.dedup` | 0.7039 | 0.8168 | 300 | vector | - | - | BAAI/bge-reranker-base | 50 |
| `scifact_rerank_msmarco_miniLM12.topk50.dedup` | 0.6968 | 0.8114 | 300 | vector | - | - | cross-encoder/ms-marco-MiniLM-L-12-v2 | 50 |
| `scifact_rerank_msmarco_miniLM12.topk100.dedup` | 0.6913 | 0.7832 | 300 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM12.topk20.dedup` | 0.6913 | 0.7832 | 300 | - | - | - | - | - |
| `scifact_hybrid_rerank_miniLM12.topk100.dedup` | 0.6889 | 0.7931 | 300 | hybrid | 0.7 | 5 | cross-encoder/ms-marco-MiniLM-L-12-v2 | 100 |
| `scifact_rerank_msmarco_miniLM6.dedup` | 0.6870 | 0.7832 | 300 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM6.topk100.dedup` | 0.6870 | 0.7832 | 300 | - | - | - | - | - |
| `scifact_rerank_msmarco_miniLM6.topk20.dedup` | 0.6870 | 0.7832 | 300 | - | - | - | - | - |
| `scifact_baseline.dedup` | 0.6406 | 0.7832 | 300 | vector | - | - | - | - |
| `scifact_baseline` | 0.6406 | 0.7832 | 300 | vector | - | - | - | - |
