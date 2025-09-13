# Hybrid Retrieval Optimization – Redo Log (Alpha Wiring + Evaluation Fixes)

This log documents code changes and the re-run procedure to correct the initial hybrid optimization experiment. The focus is wiring the alpha parameter (similarity_threshold) end-to-end, recording it in results, preferring retrieval-quality metrics when available, and making the pipeline reproducible/headless-friendly.

## What Changed

- Alpha wiring
  - `src/retriever.py`: `retrieve(..., alpha: Optional[float])` and pass through to `_hybrid_retrieve`.
  - `src/rag_pipeline.py`: `query(..., similarity_threshold: Optional[float])` and forward to retriever only for `retrieval_method='hybrid'`. Also echo `similarity_threshold` in response metadata.
  - `src/experiment_runner.py`:
    - Pass `retrieval_method` and `similarity_threshold` from `ExperimentConfig` to `RAGPipeline.query`.
    - Support queries with `{id, query}`; preserve `query_id` in run metrics.
    - Save retrieved doc IDs under `metrics.retrieved_doc_ids` for later evaluation.
  - `main.py` (`experiment sweep`): If sweeping `similarity_threshold`, set base retrieval method to `hybrid` by default.
  - `main.py` (`_save_experiment_results`): Persist `similarity_threshold` in the saved config block.

- Analysis & visualization
  - `experiments/hybrid/analyze_results.py`:
    - Group strictly by explicit `similarity_threshold`.
    - Prefer `ndcg@10`/`recall@10` if present; fallback to response time.
  - `experiments/hybrid/visualize_results.py`:
    - Robust alpha-key handling (string vs float) and headless mode (`MPL_HEADLESS=1`).
    - Avoid blocking `plt.show()` in headless environments.

- Portability and headless safety
  - `experiments/hybrid/monitor_and_analyze.py`: Remove hard-coded `chdir`.

- Query subsets
  - Added `test_data/fiqa_subset_queries.json` and `test_data/scifact_subset_queries.json` with stable `{id, query}` pairs (20 queries each) for quick verification sweeps.

## Re-run Procedure

Environment
- Always run with the project conda env prefix:
  `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python`

Verification sweeps (quick sanity)
- FiQA (α in {0.0, 0.5, 1.0}):
  ```bash
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
  python main.py experiment sweep \
    --param similarity_threshold \
    --values "0.0,0.5,1.0" \
    --queries test_data/fiqa_subset_queries.json \
    --corpus fiqa_technical \
    --output experiments/hybrid/results/fiqa_alpha_verify.json
  ```
- SciFact (α in {0.0, 0.5, 1.0}):
  ```bash
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
  python main.py experiment sweep \
    --param similarity_threshold \
    --values "0.0,0.5,1.0" \
    --queries test_data/scifact_subset_queries.json \
    --corpus scifact_scientific \
    --output experiments/hybrid/results/scifact_alpha_verify.json
  ```

Full sweeps (11 α values if time permits)
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py experiment sweep \
  --param similarity_threshold \
  --range "0.0,1.0,0.1" \
  --queries test_data/fiqa_subset_queries.json \
  --corpus fiqa_technical \
  --output experiments/hybrid/results/fiqa_alpha_sweep_fixed.json

source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py experiment sweep \
  --param similarity_threshold \
  --range "0.0,1.0,0.1" \
  --queries test_data/scifact_subset_queries.json \
  --corpus scifact_scientific \
  --output experiments/hybrid/results/scifact_alpha_sweep_fixed.json
```

Analysis & figures
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python experiments/hybrid/analyze_results.py
MPL_HEADLESS=1 python experiments/hybrid/visualize_results.py
```

## Notes
- Retrieval-quality (NDCG/Recall) requires ground-truth qrels and query IDs aligned with BEIR. The added subsets preserve `id`; if you provide aligned qrels for these IDs, `analyze_results.py` will automatically prefer quality metrics.
- Artifacts from the first run that lacked alpha in results are left intact for reference. New outputs use `*_fixed.json` filenames.

## Quality Evaluation (BEIR-aligned)

Local sources in this repo:
- Queries: `corpus/technical/fiqa/queries.jsonl`, `corpus/narrative/scifact/queries.jsonl`
- Qrels: `corpus/technical/fiqa/qrels/test.tsv`, `corpus/narrative/scifact/qrels/test.tsv`

Prep steps:
```bash
# Convert local BEIR queries to JSON (ids preserved)
python experiments/hybrid/tools/prepare_beir_queries_local.py

# Create test-only subsets intersecting qrels (100 each)
python experiments/hybrid/tools/make_query_subset_from_qrels.py \
  --queries test_data/beir_fiqa_queries.json \
  --qrels test_data/beir_fiqa_qrels_test.tsv \
  --out test_data/fiqa_queries_subset_test_100.json \
  --n 100 --seed 42

python experiments/hybrid/tools/make_query_subset_from_qrels.py \
  --queries test_data/beir_scifact_queries.json \
  --qrels test_data/beir_scifact_qrels_test.tsv \
  --out test_data/scifact_queries_subset_test_100.json \
  --n 100 --seed 42
```

Verification sweeps + quality augmentation:
```bash
# FiQA (100 test queries x 3 alphas), retrieval-only
RAG_SWEEP_NO_LLM=1 python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/fiqa_queries_subset_test_100.json \
  --corpus fiqa_technical \
  --output experiments/hybrid/results/fiqa_alpha_beir_test100.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset fiqa \
  --results experiments/hybrid/results/fiqa_alpha_beir_test100.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/fiqa_alpha_beir_test100.quality.json

# SciFact (100 test queries x 3 alphas), retrieval-only
RAG_SWEEP_NO_LLM=1 python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/scifact_queries_subset_test_100.json \
  --corpus scifact_scientific \
  --output experiments/hybrid/results/scifact_alpha_beir_test100.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset scifact \
  --results experiments/hybrid/results/scifact_alpha_beir_test100.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/scifact_alpha_beir_test100.quality.json

# Point analysis at enriched files
cp experiments/hybrid/results/*beir_test100.quality.json \
   experiments/hybrid/results/{fiqa,scifact}_alpha_sweep.json
python experiments/hybrid/analyze_results.py
MPL_HEADLESS=1 python experiments/hybrid/visualize_results.py
```

Observed on 100‑query test subsets:
- FiQA: avg NDCG@10 ≈ 0.300, Recall@10 ≈ 0.327 for α ∈ {0.0, 0.5, 1.0}
- SciFact: avg NDCG@10 ≈ 0.783, Recall@10 ≈ 0.807 for α ∈ {0.0, 0.5, 1.0}
- Invariance across α in this slice suggests fusion and/or candidate pool require tuning to expose α sensitivity.

Next steps:
- Increase candidate pools (e.g., collect 5×k per method before fusion) and/or adjust normalization.
- Try alternative fusion (e.g., z‑score, RRF) and re‑eval.
- Expand to full test sets for both datasets.

## Change: Larger Hybrid Candidate Pools (5×k)

Implementation:
- `src/vector_database.py`: `hybrid_search(..., candidate_multiplier=5)` and use `fetch_n = k*candidate_multiplier` for each method before fusion.
- `src/retriever.py`: `retrieve(..., candidate_multiplier)` → `_hybrid_retrieve` → `vector_db.hybrid_search(..., candidate_multiplier)`; supports env override `RAG_HYBRID_CAND_MULT`.
- `src/rag_pipeline.py`: `query(..., candidate_multiplier)` forwards to retriever for hybrid method.

Commands run:
```bash
RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/fiqa_queries_subset_test_100.json \
  --corpus fiqa_technical \
  --output experiments/hybrid/results/fiqa_alpha_beir_test100_x5.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset fiqa \
  --results experiments/hybrid/results/fiqa_alpha_beir_test100_x5.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/fiqa_alpha_beir_test100_x5.quality.json

RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/scifact_queries_subset_test_100.json \
  --corpus scifact_scientific \
  --output experiments/hybrid/results/scifact_alpha_beir_test100_x5.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset scifact \
  --results experiments/hybrid/results/scifact_alpha_beir_test100_x5.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/scifact_alpha_beir_test100_x5.quality.json
```

Outcome (100‑query test subsets, 5×k candidates):
- FiQA: α ∈ {0.0, 0.5, 1.0} → avg NDCG@10 ≈ 0.300, Recall@10 ≈ 0.327 (no change)
- SciFact: α ∈ {0.0, 0.5, 1.0} → avg NDCG@10 ≈ 0.783, Recall@10 ≈ 0.807 (no change)

Interpretation: With current normalization (per‑list max) and these subsets, α remains quality‑insensitive even with larger candidate pools. Next, test alternative fusions (e.g., z‑score normalization, reciprocal rank fusion) and expand to full test sets.

## Change: Alternative Fusion Methods (z‑score)

Implementation:
- `src/vector_database.py`: hybrid fusion supports `fusion_method` = `zscore` | `rrf` | `maxnorm` (default). Z‑score uses per‑list z‑scores; combined = α*z_vec + (1‑α)*z_kw. RRF uses α*(1/(K+rank_vec)) + (1‑α)*(1/(K+rank_kw)), default K=60.
- `src/retriever.py`: reads env `RAG_HYBRID_FUSION` and `RAG_HYBRID_RRF_K` and forwards to vector DB.

Commands (100 test queries, 3 α, 5×k candidates):
```bash
RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 RAG_HYBRID_FUSION=zscore \
  python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/fiqa_queries_subset_test_100.json \
  --corpus fiqa_technical \
  --output experiments/hybrid/results/fiqa_alpha_beir_test100_x5_z.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset fiqa \
  --results experiments/hybrid/results/fiqa_alpha_beir_test100_x5_z.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/fiqa_alpha_beir_test100_x5_z.quality.json

RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 RAG_HYBRID_FUSION=zscore \
  python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/scifact_queries_subset_test_100.json \
  --corpus scifact_scientific \
  --output experiments/hybrid/results/scifact_alpha_beir_test100_x5_z.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset scifact \
  --results experiments/hybrid/results/scifact_alpha_beir_test100_x5_z.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/scifact_alpha_beir_test100_x5_z.quality.json
```

Observed (100‑query test subsets, 5×k, z‑score fusion):
- FiQA: α=0.0 → NDCG@10≈0.111, α=0.5→0.300, α=1.0→0.300; Recall mirrors NDCG (α=0.5/1.0 ≈0.327).
- SciFact: α=0.0 → NDCG@10≈0.110, α=0.5→0.783, α=1.0→0.783; Recall@10 highest at α≥0.5 (≈0.807).

Interpretation: z‑score fusion reveals expected α sensitivity on these subsets (keyword‑only underperforms; α≥0.5 best). Next: RRF fusion and expansion to full test sets for stronger evidence.

## Change: Alternative Fusion Methods (RRF)

Implementation:
- `src/vector_database.py`: adds `fusion_method='rrf'` path with `rrf_k` (default 60).
- `src/retriever.py`: reads env `RAG_HYBRID_FUSION=rrf` and `RAG_HYBRID_RRF_K`.

Commands (100 test queries, 3 α, 5×k candidates):
```bash
RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 RAG_HYBRID_FUSION=rrf RAG_HYBRID_RRF_K=60 \
  python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/fiqa_queries_subset_test_100.json \
  --corpus fiqa_technical \
  --output experiments/hybrid/results/fiqa_alpha_beir_test100_x5_rrf.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset fiqa \
  --results experiments/hybrid/results/fiqa_alpha_beir_test100_x5_rrf.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/fiqa_alpha_beir_test100_x5_rrf.quality.json

RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 RAG_HYBRID_FUSION=rrf RAG_HYBRID_RRF_K=60 \
  python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/scifact_queries_subset_test_100.json \
  --corpus scifact_scientific \
  --output experiments/hybrid/results/scifact_alpha_beir_test100_x5_rrf.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset scifact \
  --results experiments/hybrid/results/scifact_alpha_beir_test100_x5_rrf.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/scifact_alpha_beir_test100_x5_rrf.quality.json
```

Observed (100‑query test subsets, 5×k, RRF fusion):
- FiQA: α=0.0 → NDCG@10≈0.070; α=0.5→0.300; α=1.0→0.300; recall mirrors NDCG (α≥0.5 best).
- SciFact: α=0.0 → NDCG@10≈0.148; α=0.5→0.783; α=1.0→0.783; Recall@10 peaks at α≥0.5 (≈0.807).

Conclusion: Both z‑score and RRF fusions expose α sensitivity consistent with expectations on these test subsets. With legacy per‑list max normalization, α differences were flattened.

### Full Test Sets (RRF, 5×k, 3 α)

Commands:
```bash
RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 RAG_HYBRID_FUSION=rrf RAG_HYBRID_RRF_K=60 \
  python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/fiqa_queries_test_all.json \
  --corpus fiqa_technical \
  --output experiments/hybrid/results/fiqa_alpha_beir_full_x5_rrf.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset fiqa \
  --results experiments/hybrid/results/fiqa_alpha_beir_full_x5_rrf.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/fiqa_alpha_beir_full_x5_rrf.quality.json

RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 RAG_HYBRID_FUSION=rrf RAG_HYBRID_RRF_K=60 \
  python main.py --db-path data/rag_vectors.db \
  experiment sweep --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/scifact_queries_test_all.json \
  --corpus scifact_scientific \
  --output experiments/hybrid/results/scifact_alpha_beir_full_x5_rrf.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset scifact \
  --results experiments/hybrid/results/scifact_alpha_beir_full_x5_rrf.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/scifact_alpha_beir_full_x5_rrf.quality.json
```

Observed (RRF, full test sets):
- FiQA (648 queries):
  - α=0.0 → NDCG@10≈0.095, Recall@10≈0.123, avg time≈0.61s
  - α=0.5 → NDCG@10≈0.342, Recall@10≈0.367, avg time≈1.68s
  - α=1.0 → NDCG@10≈0.342, Recall@10≈0.367, avg time≈0.49s
- SciFact (300 queries):
  - α=0.0 → NDCG@10≈0.127, Recall@10≈0.176, avg time≈0.13s
  - α=0.5 → NDCG@10≈0.722, Recall@10≈0.740, avg time≈0.11s
  - α=1.0 → NDCG@10≈0.722, Recall@10≈0.740, avg time≈0.11s

Conclusion: On full test sets, RRF again favors α≥0.5 for quality. For FiQA, α=1.0 is as good as α=0.5 in quality and faster on average. For SciFact, α=0.5 and α=1.0 are tied in quality and latency.
