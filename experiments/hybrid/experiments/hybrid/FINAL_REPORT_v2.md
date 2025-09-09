# Hybrid Retrieval Optimization with BEIR (Revisited): Alpha, Fusion, and Candidate Pooling

## Abstract
We revisit hybrid retrieval optimization for a local RAG system using BEIR test sets (FiQA and SciFact). We correct the alpha (similarity_threshold) wiring, introduce principled fusion strategies (z‑score and reciprocal rank fusion, RRF), and scale candidate pools to improve sensitivity to alpha. We evaluate quality using BEIR qrels (NDCG@10, Recall@10) and measure latency across 3 alpha settings (0.0, 0.5, 1.0). On full test sets, both z‑score and RRF reveal clear alpha sensitivity consistent with expectations: α ≥ 0.5 is optimal for retrieval quality across datasets. Under RRF on FiQA, α=1.0 matches α=0.5 in quality and is faster, suggesting a domain‑specific throughput advantage. We provide complete reproducibility steps and release all artifacts.

## 1. Introduction
Hybrid retrieval combines dense (vector) and sparse (keyword) signals. Prior runs were inconclusive due to missing alpha provenance and an analysis that optimized for response time rather than retrieval quality. We re‑implemented the pipeline to (1) correctly wire alpha end‑to‑end, (2) record per‑run configs and retrieved doc IDs, (3) compute BEIR‑aligned quality metrics, and (4) introduce robust fusion and pooling to expose alpha effects.

## 2. Datasets and Ground Truth
- FiQA (BEIR FiQA‑2018): 648 test queries with qrels
- SciFact: 300 test queries with qrels

We converted `corpus/*/*/queries.jsonl` to `test_data/beir_{dataset}_queries.json` and intersected with test qrels to produce `test_data/*_queries_test_all.json`. This preserves BEIR query IDs for quality evaluation.

## 3. System and Corrections
- Alpha wiring: `similarity_threshold` now drives hybrid α and is recorded in result configs.
- Fusion methods (in `vector_database.hybrid_search`):
  - MaxNorm: per‑list max normalization (legacy)
  - Z‑Score: per‑list z‑score, combined score = α*z_vec + (1−α)*z_kw
  - RRF: combined score = α*(1/(K+rank_vec)) + (1−α)*(1/(K+rank_kw)), K=60
- Candidate pooling: fetch 5×k candidates from each method before fusion to avoid early truncation.
- Retrieval‑only sweeps: `RAG_SWEEP_NO_LLM=1` for speed and purity of retrieval evaluation.
- Quality augmentation: computed NDCG@10 and Recall@10 using BEIR qrels and DB mapping (doc_uuid → BEIR doc ID via `source_path`).

## 4. Experimental Design
- Alpha settings: α ∈ {0.0, 0.5, 1.0}
- k=5, candidate_multiplier=5 (total 25 candidates per method → fused to top‑k)
- Fusion methods: z‑score and RRF (primary)
- Metrics: NDCG@10, Recall@10, average per‑query latency
- Datasets: FiQA (648 test queries), SciFact (300 test queries)

## 5. Results (Full Test Sets)
### 5.1 Z‑Score Fusion (5×k)
- FiQA (648):
  - α=0.0 → NDCG@10≈0.089; Recall@10≈0.113; Time≈0.72s
  - α=0.5 → NDCG@10≈0.342; Recall@10≈0.367; Time≈0.73s
  - α=1.0 → NDCG@10≈0.342; Recall@10≈0.367; Time≈3.49s
  - Optimal α (quality): 0.50 (α=1.0 ties in quality but slower)
- SciFact (300):
  - α=0.0 → NDCG@10≈0.124; Recall@10≈0.198; Time≈0.18s
  - α=0.5 → NDCG@10≈0.722; Recall@10≈0.740; Time≈0.16s
  - α=1.0 → NDCG@10≈0.722; Recall@10≈0.740; Time≈0.16s
  - Optimal α (quality): 0.50 (ties with α=1.0 in quality and latency)

### 5.2 RRF Fusion (5×k)
- FiQA (648):
  - α=0.0 → NDCG@10≈0.095; Recall@10≈0.123; Time≈0.61s
  - α=0.5 → NDCG@10≈0.342; Recall@10≈0.367; Time≈1.68s
  - α=1.0 → NDCG@10≈0.342; Recall@10≈0.367; Time≈0.49s
  - Optimal α (quality): α=0.5 and α=1.0 tie; α=1.0 faster → preferred for throughput
- SciFact (300):
  - α=0.0 → NDCG@10≈0.127; Recall@10≈0.176; Time≈0.13s
  - α=0.5 → NDCG@10≈0.722; Recall@10≈0.740; Time≈0.11s
  - α=1.0 → NDCG@10≈0.722; Recall@10≈0.740; Time≈0.11s

## 6. Comparison: Z‑Score vs RRF
- Quality: Both fusions agree α≥0.5 is best on FiQA and SciFact.
- Latency: On FiQA, z‑score favors α=0.5 over α=1.0 due to a long tail for α=1.0; RRF in contrast shows α=1.0 as the fastest among quality‑optimal α. On SciFact, α=0.5 and α=1.0 tie in both fusions.
- Practical default: α=0.5 remains the best universal default across datasets.
- Domain‑specific knob: For FiQA under RRF, α=1.0 attains the same quality and better latency; this is a viable speed‑tuned setting.

## 7. Recommendations
- Default hybrid configuration: α=0.5, z‑score or RRF fusion, candidate_multiplier=5.
- Throughput‑tuned FiQA (RRF): α=1.0 for same quality and lower latency.
- Implement a runtime flag for fusion selection and candidate multiplier (e.g., CLI options rather than env) to ease experimentation and production rollout.
- Consider a simple dynamic policy: α=1.0 for short keyword‑heavy FiQA‑style queries; α=0.5 otherwise.

## 8. Threats to Validity / Limitations
- Fusion dependence: α sensitivity depends on the choice of fusion; maxnorm can mask differences.
- Latency variability: system state and caching can affect timings; reported averages are informative but environment‑specific.
- Retrieval@k only: NDCG@10/Recall@10 reflect first‑stage retrieval, not end‑to‑end answer faithfulness.
- RRF K parameter: set to 60; alternative K may adjust results slightly.

## 9. Reproducibility
### Environment
- Conda env: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`
- Database: `data/rag_vectors.db` (includes fiqa_technical and scifact_scientific)

### Data Prep
```bash
python experiments/hybrid/tools/prepare_beir_queries_local.py
python experiments/hybrid/tools/make_query_subset_from_qrels.py \
  --queries test_data/beir_fiqa_queries.json \
  --qrels test_data/beir_fiqa_qrels_test.tsv \
  --out test_data/fiqa_queries_test_all.json --n 100000
python experiments/hybrid/tools/make_query_subset_from_qrels.py \
  --queries test_data/beir_scifact_queries.json \
  --qrels test_data/beir_scifact_qrels_test.tsv \
  --out test_data/scifact_queries_test_all.json --n 100000
```

### Sweeps (examples)
Z‑score fusion, full test sets (3 α):
```bash
RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 RAG_HYBRID_FUSION=zscore \
  python main.py --db-path data/rag_vectors.db experiment sweep \
  --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/fiqa_queries_test_all.json \
  --corpus fiqa_technical \
  --output experiments/hybrid/results/fiqa_alpha_beir_full_x5_z.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset fiqa \
  --results experiments/hybrid/results/fiqa_alpha_beir_full_x5_z.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/fiqa_alpha_beir_full_x5_z.quality.json
```
RRF fusion, full test sets (3 α):
```bash
RAG_SWEEP_NO_LLM=1 RAG_HYBRID_CAND_MULT=5 RAG_HYBRID_FUSION=rrf RAG_HYBRID_RRF_K=60 \
  python main.py --db-path data/rag_vectors.db experiment sweep \
  --param similarity_threshold --values "0.0,0.5,1.0" \
  --queries test_data/fiqa_queries_test_all.json \
  --corpus fiqa_technical \
  --output experiments/hybrid/results/fiqa_alpha_beir_full_x5_rrf.json

python experiments/hybrid/tools/augment_results_with_quality.py \
  --dataset fiqa \
  --results experiments/hybrid/results/fiqa_alpha_beir_full_x5_rrf.json \
  --db data/rag_vectors.db \
  --out experiments/hybrid/results/fiqa_alpha_beir_full_x5_rrf.quality.json
```

### Analysis & Figures
```bash
# Point analyzer to enriched files (copy or set expected names)
cp experiments/hybrid/results/fiqa_alpha_beir_full_x5_z.quality.json \
   experiments/hybrid/results/fiqa_alpha_sweep.json
cp experiments/hybrid/results/scifact_alpha_beir_full_x5_z.quality.json \
   experiments/hybrid/results/scifact_alpha_sweep.json

python experiments/hybrid/analyze_results.py
MPL_HEADLESS=1 python experiments/hybrid/visualize_results.py
```

## 10. Time and Resource Footprint
- Z‑score full test sets (observed): FiQA ~71 min; SciFact ~3.8 min (~75 min total)
- RRF full test sets (observed): FiQA ~35–55 min; SciFact ~2–4 min (~60–80 min total)
- Combined (both fusions): ~2–2.5 hours; conservatively budget ~3 hours to account for variability.
- Hardware: Apple Silicon (MPS) with sqlite‑vec; retrieval‑only (no LLM generation).

## 11. Conclusions
This study demonstrates that, with correct alpha wiring and robust fusion/pooling, hybrid retrieval quality on BEIR FiQA and SciFact is consistently optimal at α ≥ 0.5. Z‑score and RRF clearly expose α sensitivity (whereas maxnorm can mask it). For FiQA, RRF suggests α=1.0 can match α=0.5 in quality while reducing latency—a valuable speed‑tuning option. We recommend a universal default of α=0.5 with z‑score or RRF, candidate_multiplier=5, and optional domain‑specific tuning (e.g., α=1.0 for FiQA under RRF when throughput matters). All code and artifacts are released to ensure reproducibility.

---
Generated: 2025‑09‑07
Artifacts: `experiments/hybrid/results/*`, `experiments/hybrid/analysis/*`, `experiments/hybrid/figures/*`
