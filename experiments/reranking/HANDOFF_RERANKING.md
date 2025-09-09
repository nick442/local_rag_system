Handoff: v2.2 Reranking Experiments (BEIR)

Objective
- Two-stage retrieval with optional CrossEncoder reranker on BEIR datasets (FiQA, SciFact), minimal and non-breaking.

Prepared
- `src/reranker_service.py`: Optional reranker (disabled by default). Enable via env `RAG_RERANKER_MODEL` or the runner flag.
- `src/rag_pipeline.py`: Rerank hook applied before prompt building when enabled.
- `experiments/reranking/run_reranking_experiment.py`: Retrieval-only runner; supports `--retrieval-method`, `--alpha`, `--candidate-multiplier`, `--reranker-model`, `--rerank-topk`.
- `experiments/reranking/evaluate_reranking.py`: Computes NDCG@K and Recall@K; aligns BEIR corpus-ids from filename/source stem.
- `experiments/Opus_proposals_v2/v2_2_reranking_BEIR.md`: Section 11 contains current results and next steps.

Environment
- Activate: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`
- DB: `data/rag_vectors.db` (collections: `fiqa_technical`, `scifact_scientific`)
- Ignore torchvision image warnings.

Preflight (IDs aligned to qrels)
- Generate test-only queries with IDs matching qrels (not committed by default):
```
# FiQA
python - << 'PY'
import json;from pathlib import Path
qrels=Path('test_data/beir_fiqa_qrels_test.tsv'); q=Path('test_data/beir_fiqa_queries.json')
qids={l.split('\t')[0] for i,l in enumerate(qrels.open()) if i>0}
data=json.loads(q.read_text()); qs=data['queries'] if 'queries' in data else data
out=[x for x in qs if str(x.get('id') or x.get('query_id') or '').strip() in qids]
Path('test_data/beir_fiqa_queries_test_only.json').write_text(json.dumps({'metadata':{'dataset':'fiqa','note':'qrels test only'},'queries':out}))
print(len(out),'fiqa queries written')
PY
# SciFact
python - << 'PY'
import json;from pathlib import Path
qrels=Path('test_data/beir_scifact_qrels_test.tsv'); q=Path('test_data/beir_scifact_queries.json')
qids={l.split('\t')[0] for i,l in enumerate(qrels.open()) if i>0}
data=json.loads(q.read_text()); qs=data['queries'] if 'queries' in data else data
out=[x for x in qs if str(x.get('id') or x.get('query_id') or '').strip() in qids]
Path('test_data/beir_scifact_queries_test_only.json').write_text(json.dumps({'metadata':{'dataset':'scifact','note':'qrels test only'},'queries':out}))
print(len(out),'scifact queries written')
PY
```

Quick Commands (k=10)
1) Baselines (vector-only)
```
# FiQA
RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
  --queries test_data/beir_fiqa_queries_test_only.json \
  --corpus fiqa_technical --k 10 \
  --output experiments/reranking/results/fiqa_baseline.test.json
# SciFact
RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
  --queries test_data/beir_scifact_queries_test_only.json \
  --corpus scifact_scientific --k 10 \
  --output experiments/reranking/results/scifact_baseline.test.json
```

2) Reranked (MiniLM-L-12-v2)
```
# FiQA
RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
  --queries test_data/beir_fiqa_queries_test_only.json \
  --corpus fiqa_technical --k 10 --rerank-topk 50 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --output experiments/reranking/results/fiqa_rerank_msmarco_miniLM12.topk50.test.json
# SciFact
RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
  --queries test_data/beir_scifact_queries_test_only.json \
  --corpus scifact_scientific --k 10 --rerank-topk 50 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --output experiments/reranking/results/scifact_rerank_msmarco_miniLM12.topk50.test.json
```

3) Hybrid first-stage (FiQA example)
```
RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
  --queries test_data/beir_fiqa_queries_test_only.json \
  --corpus fiqa_technical --k 10 \
  --retrieval-method hybrid --alpha 0.7 --candidate-multiplier 5 \
  --rerank-topk 100 --reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --output experiments/reranking/results/fiqa_hybrid_rerank_miniLM12.topk100.test.json
```

4) Stronger reranker (if network/time allow)
```
# BGE base (FiQA example)
RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
  --queries test_data/beir_fiqa_queries_test_only.json \
  --corpus fiqa_technical --k 10 --rerank-topk 100 \
  --reranker-model BAAI/bge-reranker-base \
  --output experiments/reranking/results/fiqa_rerank_bge_base.topk100.test.json
```

Evaluate (NDCG@10 and Recall@10)
```
# Replace paths as needed
python experiments/reranking/evaluate_reranking.py \
  --results experiments/reranking/results/fiqa_baseline.test.json \
  --qrels test_data/beir_fiqa_qrels_test.tsv --k 10 \
  --output experiments/reranking/results/fiqa_baseline.metrics.json
```

Deliverables
- Commit small JSONs only (results/metrics) to `experiments/reranking/results/`.
- Update `experiments/Opus_proposals_v2/v2_2_reranking_BEIR.md` Section 11 with a brief summary per batch.
- Add a concise PR #18 comment with ndcg@10_mean and recall@10_mean deltas.

Known Pitfalls
- Queries must have IDs; use the test-only query files above.
- Evaluator relies on filename/source stem for corpus-id mapping to qrels.
- Hybrid may emit FTS5 warnings; safe to ignore if results are written.
- torchvision image warnings are safe to ignore.

Next Steps Checklist
- [ ] Run BGE Base for FiQA and SciFact (`--rerank-topk 50/100`), evaluate and compare.
- [ ] FiQA hybrid tuning: sweep `--alpha {0.3,0.5,0.7,0.9}` × `--candidate-multiplier {3,5,10}` with L‑12/BGE.
- [ ] Keep SciFact first-stage `vector`; try stronger reranker only.
- [ ] Add TREC-COVID once queries and qrels exist in `test_data/`.

