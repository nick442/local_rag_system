# Enhanced RAG System — Capabilities and Usage Guide

This guide explains the full capabilities of the enhanced codebase and how to use them day‑to‑day. It covers core CLI workflows, retrieval and hybrid search, reranking, experiments, evaluation, packaging finished experiments, environment tips, and troubleshooting.

All commands assume the project conda environment:

```
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python
```

When examples show `python ...`, always prefix with the activation line above (or export an alias).

## 1) Quick Start

- Check system status (DB, collections, models):
  - `python main.py --db-path data/rag_vectors.db status`

- Ingest sample data (dry run):
  - `python main.py ingest directory sample_corpus --collection demo --dry-run`

- Query once (vector retrieval, default k):
  - `python main.py query "What is machine learning?" --collection demo`

- List collections:
  - `python main.py collection list`

Tip: For all experiments and non‑LLM runs, set `RAG_SWEEP_NO_LLM=1` to avoid loading generation models:

```
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
RAG_SWEEP_NO_LLM=1 python main.py --db-path data/rag_vectors.db status
```

## 2) Core CLI (main.py)

Key commands (abbreviated):

- Status and diagnostics
  - `python main.py status` — DB and collection summary

- Collection management
  - `python main.py collection create --name fiqa_technical`
  - `python main.py collection list`
  - `python main.py collection switch fiqa_technical`
  - `python main.py collection delete fiqa_technical`
  - `python main.py collection export fiqa_technical --output backups/fiqa_export.json`

- Ingest
  - Directory: `python main.py ingest directory <path> --collection <id> [--pattern *.md] [--max-workers 4]`
  - Single file: `python main.py ingest file <file> --collection <id>`

- Analysis and quality
  - `python main.py analyze collection --collection <id>`
  - `python main.py analyze quality --collection <id>`
  - `python main.py analyze export-report --collection <id> --output reports/quality.json`

- Maintenance
  - `python main.py deduplicate --collection <id> [--dry-run]`
  - `python main.py reindex --collection <id> --operation [rebuild|reembed|rechunk|vacuum] [--backup]`
  - `python main.py validate-integrity [--collection <id>]`

- Query/Chat
  - `python main.py query "<question>" --collection <id> [--k 10] [--metrics]`
  - `python main.py chat --collection <id> [--profile fast] [--no-streaming]`

- Batch experiments (simple)
  - `python main.py experiment batch --queries <json> --collection <id> --k 10 --output results/run.json [--dry-run]`

Notes:
- Default paths live in `main.py` and can be overridden by flags.
- Logs are written under `logs/` using existing helpers.

## 3) Retrieval Backends

Three first‑stage retrieval methods are supported in the pipeline and experiment runner:

- `vector` — ANN search via sqlite‑vec (`float[384]`) on the embeddings table.
- `keyword` — FTS5 keyword search on `chunks_fts` (BM25‑like scoring).
- `hybrid` — Fuses vector and keyword lists with pluggable methods:
  - `maxnorm` (default): per‑list max normalization + weighted sum
  - `zscore`: z‑score normalization then weighted sum
  - `rrf`: Reciprocal Rank Fusion (RRF‑K)

Hybrid parameters:
- `alpha` — weight for vector scores (1‑alpha for keyword)
- `candidate‑multiplier` (cm) — fetch `k*cm` from each method before fusing
- `fusion_method` — via env `RAG_HYBRID_FUSION` in code paths using env (default `maxnorm`)
- `RAG_HYBRID_RRF_K` — RRF hyper‑parameter (default 60)

Environment overrides (optional):
- `RAG_HYBRID_CAND_MULT` (fallback cm), `RAG_HYBRID_FUSION`, `RAG_HYBRID_RRF_K`

Vector DB notes:
- Embedding dimension is validated on open (384). Mismatches raise a clear error.
- sqlite‑vec extension loads via Python package if available, else falls back to vendor dylib under `vendor/sqlite-vec/vec0.dylib` (unless disabled with `RAG_DISABLE_SQLITE_VEC_VENDOR=1`).

## 4) Reranking (CrossEncoder)

The pipeline supports an optional reranking step via `src/reranker_service.py` and `src/rag_pipeline.py`.

- Enable by setting the env var or passing a flag (experiment runner):
  - `RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2`
  - or `--reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2`
- Common models:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (fastest)
  - `cross-encoder/ms-marco-MiniLM-L-12-v2` (balanced)
  - `BAAI/bge-reranker-base` (strongest in many settings)
- Control reranked pool size with `--rerank-topk` (e.g., 50 or 100).

Performance tips:
- Larger rerank‑topk increases latency. Start with 50 when evaluating at k=10.
- Device auto‑selects `mps` (Apple Silicon) or CPU; adjust if needed in `reranker_service`.

## 5) Reranking Experiment Runner

Script: `experiments/reranking/run_reranking_experiment.py`

Usage:
```
RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py \
  --queries <queries.json> \
  --corpus <collection_id> \
  --k 10 \
  [--db-path data/rag_vectors.db] \
  [--retrieval-method vector|keyword|hybrid] \
  [--alpha 0.5] [--candidate-multiplier 5] \
  [--reranker-model <hf_model>] [--rerank-topk 50] \
  --output experiments/reranking/results/<name>.test.json
```

Examples:
- FiQA baseline (vector only):
  - `... --queries test_data/beir_fiqa_queries_test_only.json --corpus fiqa_technical --k 10 --output experiments/reranking/results/fiqa_baseline.test.json`
- FiQA hybrid + MiniLM‑L12:
  - `... --retrieval-method hybrid --alpha 0.5 --candidate-multiplier 5 --rerank-topk 50 --reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2 --output experiments/reranking/results/fiqa_hybrid_a0.5_cm5_miniLM12.topk50.test.json`
- SciFact BGE (vector):
  - `... --queries test_data/beir_scifact_queries_test_only.json --corpus scifact_scientific --k 10 --rerank-topk 50 --reranker-model BAAI/bge-reranker-base --output experiments/reranking/results/scifact_rerank_bge_base.topk50.test.json`

Notes:
- The runner automatically inflates retrieve‑k so reranking has enough candidates, then truncates contexts to final `k`.

## 6) Evaluation

Script: `experiments/reranking/evaluate_reranking.py`

Maps BEIR corpus IDs from `metadata.filename`/`metadata.source` stem (e.g., `12345.txt → 12345`). Requires query IDs.

```
python experiments/reranking/evaluate_reranking.py \
  --results experiments/reranking/results/<name>.test.json \
  --qrels test_data/beir_<dataset>_qrels_test.tsv \
  --k 10 \
  --output experiments/reranking/results/<name>.metrics.json \
  [--deduplicate]
```

Outputs aggregate `ndcg@k_mean`, `recall@k_mean`, and per‑query metrics. Use `--deduplicate` to remove repeated doc IDs before scoring.

## 7) Summaries, Reporting, and Packaging

- Summarize all metrics and produce a Markdown table:
  - `python experiments/reranking/summarize_results.py --results-dir experiments/reranking/results --output experiments/reranking/results/summary.json --markdown experiments/reranking/results/summary.md`

- Generate a final report:
  - `python experiments/reranking/generate_final_report.py --summary-json experiments/reranking/results/summary.json --output experiments/reranking/FINAL_REPORT_v2_2_BEIR.md`

- Package a finished experiment into the catalog:
  - `scripts/experiments/finalize_to_catalog.sh beir_v2_2_reranking`
  - Creates `experiments/finished/beir_v2_2_reranking/` with MANIFEST, results, reports, paper, scripts, env, logs, and updates `experiments/catalog.json`.

Convenience scripts:
- `scripts/tests/run_summary.sh` — refresh summary + final report
- `scripts/tests/monitor_rerank_jobs.sh` — live job/metrics progress view
- `scripts/tests/finalize_reranking.sh` — waits for jobs, regenerates, updates proposal + PR note

## 8) Environment and Configuration

- Activate env: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`
- Disable LLM: `RAG_SWEEP_NO_LLM=1`
- Reranker: `RAG_RERANKER_MODEL=<hf_name>` or pass `--reranker-model` in the runner
- Hybrid fusion env (optional): `RAG_HYBRID_CAND_MULT`, `RAG_HYBRID_FUSION`, `RAG_HYBRID_RRF_K`
- sqlite‑vec vendor control: `RAG_DISABLE_SQLITE_VEC_VENDOR=1` to force off the vendor dylib fallback
- Model paths can be overridden by CLI flags; certain tools read `LLM_MODEL_PATH`

## 9) Performance Tips

- For k=10 evaluation, start with `--rerank-topk 50` and increase only if Recall@10 is low.
- Hybrid often helps when vector misses sparse‑lexical matches; try `alpha ∈ {0.3,0.5,0.7}` with `cm=5`.
- Use Apple Silicon `mps` via torch when available for rerankers; CPU is fine for small top‑k.
- Use temp DBs for tests; keep production DBs under `data/` and back up before `reindex`.

## 10) Troubleshooting

- Metrics are zero → Ensure you used test‑only queries with IDs; evaluator requires `query_id` and filename‑based corpus IDs.
- `FTS5 search failed: no such column: f` → Benign warning from keyword search; safe to ignore if results are written.
- sqlite‑vec load warnings → The system falls back to vendor dylib automatically unless disabled; vector search still works.
- Embedding dimension mismatch → Rebuild or reindex with the matching `embedding_dimension`.
- Large files in GitHub warnings → Commit only small JSONs (metrics/summary); use LFS or compression for large `.test.json` if needed.

## 11) Provenance and Artifacts

- Finished package for this experiment: `experiments/finished/beir_v2_2_reranking/`
  - `MANIFEST.yaml` — metadata, parameters, best runs
  - `results/` — metrics + summary
  - `reports/` — final report + PR note; paper under `reports/paper/`
  - `scripts/`, `env/`, `logs/` — reproducibility material

For an overview of all finished experiments, see `experiments/README.md` and `experiments/catalog.json`.

