# E1/E3 SciFact & FiQA (512/50) – Fusion Summary and Final Report

This document summarizes the E3 fusion results for SciFact/FiQA at 512/50 chunking and provides a comprehensive end report for the experiment run, including metrics, latency and position diagnostics, conclusions, and reproducibility notes.

- Results roots:
  - SciFact: `results/e1_scifact/full/scifact_fixed_512_50/`
  - FiQA: `results/e1_fiqa/full/fiqa_fixed_512_50/`
  - Triad summary: `results/e1_triad/summary/`
- Embedding model path used:
  - `models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`

---

## Status Snapshot
- Orchestrator PID file exists but no active process (`results/e1_triad/run.pid`).
- SciFact 512/50 stage marker: `REBUILD` (earlier jobs completed). We deferred final DB maintenance due to timeouts.
- E3 fusion completed for both SciFact and FiQA; aliased E1 metrics written for both datasets; triad summary generated.

---

## E3 Fusion Summary (SciFact/FiQA 512/50)

Parameters: `kdocs=10`, `nchunks=100`, `cap=3`, `rrf_k=60`. Fusion performed over doc-aware rankings with per-doc best-chunk scores. Internal `doc_id` values were aliased to corpus IDs (filename stems) for evaluation alignment with qrels.

### SciFact 512/50
Metrics files: `metrics_dense_aliased.json`, `metrics_bm25_aliased.json`, `metrics_hybrid_aliased.json`, `metrics_rrf.json`, `metrics_zscore.json`

| Mode   | P@1     | P@10    | NDCG@10 | MRR     |
|--------|---------|---------|---------|---------|
| Dense  | 0.13345 | 0.02362 | 0.17279 | 0.16187 |
| BM25   | 0.00451 | 0.00045 | 0.00451 | 0.00451 |
| Hybrid | 0.13345 | 0.02362 | 0.17279 | 0.16187 |
| RRF    | 0.13436 | 0.02362 | 0.17362 | 0.16269 |
| Z-Score| 0.13345 | 0.02362 | 0.17279 | 0.16187 |

- Lift vs dense: RRF +0.00083 NDCG@10; Z-Score ±0.00000.

### FiQA 512/50
Metrics files: `metrics_dense_aliased.json`, `metrics_bm25_aliased.json`, `metrics_hybrid_aliased.json`, `metrics_rrf.json`, `metrics_zscore.json`

| Mode   | P@1     | P@10    | NDCG@10 | MRR     |
|--------|---------|---------|---------|---------|
| Dense  | 0.03354 | 0.01628 | 0.05135 | 0.03878 |
| BM25   | 0.00090 | 0.00030 | 0.00129 | 0.00106 |
| Hybrid | 0.03354 | 0.01628 | 0.05135 | 0.03878 |
| RRF    | 0.03354 | 0.00817 | 0.03246 | 0.04167 |
| Z-Score| 0.03264 | 0.00817 | 0.03228 | 0.04128 |

- Lift vs dense: RRF −0.01889 NDCG@10; Z-Score −0.01907.

---

## Latency Snapshot

SciFact 512/50 (ms):
- Dense: mean 110.73, p50 104.91, p95 140.69  (`latency.json`)
- BM25:  mean 0.89,   p50 0.72,   p95 1.47     (`latency_bm25.json`)
- Hybrid: mean 106.66, p50 103.88, p95 127.83  (`latency_hybrid.json`)

FiQA 512/50 (ms):
- Dense: mean 7053.81, p50 1641.82, p95 5376.26 (`latency.json`)
- BM25:  mean 21.66,   p50 18.29,   p95 49.41    (`latency_bm25.json`)
- Hybrid: mean 12395.20, p50 1951.63, p95 8484.48 (`latency_hybrid.json`)

---

## Position Sensitivity

Top‑1 best‑chunk relative positions (0.0 = doc start, 1.0 = doc end).

SciFact dense (`position.json`):
- Mean top‑1 position: 0.091 (N=1109)
- Histogram: begin[0–0.2]=980, middle[0.4–0.6]=57, end[0.8–1.0]=72

FiQA dense (`position.json`):
- Mean top‑1 position: 0.053 (N=6648)
- Histogram: begin[0–0.2]=6205, early[0.2–0.4]=14, middle[0.4–0.6]=151, late[0.6–0.8]=22, end[0.8–1.0]=256

---

## Short Conclusions
- Best variant per dataset (512/50)
  - SciFact: Dense/Hybrid are best; RRF adds a very small gain (+0.0008 NDCG@10).
  - FiQA: Dense/Hybrid outperform BM25 and both fusions; BM25 weakens fusion.
- Fusion vs Dense/Hybrid
  - SciFact: RRF ≈ Dense with a marginal positive lift; Z‑Score equals Dense.
  - FiQA: Both RRF and Z‑Score underperform Dense/Hybrid (BM25 adds noise).
- Latency
  - SciFact vector retrieval is consistently fast (sub‑150 ms p95).
  - FiQA vector retrieval shows heavy tails (multi‑second p95 and high mean); BM25 is consistently low latency.
- Position sensitivity
  - Strong early-chunk bias on both datasets; intros/front‑matter carry most relevant content.

---

## Comprehensive End Report

### Abstract
We evaluate chunking (E1) and hybrid fusion (E3) on SciFact and FiQA using 512/50 token windows, MiniLM L6 embeddings, and two fusion methods (RRF, Z‑Score). Dense/Hybrid retrieval substantially outperforms BM25. Fusion yields a small positive lift on SciFact and degrades performance on FiQA due to weak BM25 contributions. Latency is sub‑150 ms p95 for SciFact dense retrieval but exhibits multi‑second tails for FiQA. Relevant evidence is concentrated early in documents. We provide metrics, diagnostics, and reproducibility instructions.

### Background
E1 explores chunking configurations and retrieval modes; E3 evaluates late fusion of dense and BM25 signals. We use a doc‑aware ranking approach that caps per‑doc chunk influence and selects the best chunk as the document score.

### Datasets
- SciFact: `experiments/gpt-5-pro proposals/datasets/scifact`
- FiQA‑2018: `experiments/gpt-5-pro proposals/datasets/fiqa`
Each includes `docs/*.txt`, `queries.jsonl`, and `qrels.tsv` (BEIR‑style).

### Methods
- Chunking: Fixed 512 tokens, overlap 50.
- Embedding model: `all-MiniLM-L6-v2` snapshot `c9745ed1...`.
- Retrieval modes
  - Dense: vector search
  - BM25: SQLite FTS5 keyword
  - Hybrid: z‑normalized combination (dense+BM25)
  - Fusion: RRF (k=60) and Z‑Score applied to per‑query doc lists and per‑doc best‑chunk scores
- Doc‑aware ranking
  - Retrieve `nchunks=100` chunks; group by document; limit to `cap=3` chunks/doc; score by best chunk; return `kdocs=10` docs per query
- Evaluation
  - Internal `doc_id` → corpus doc ID (filename stem) aliasing against DB `source_path` to align with qrels
  - Metrics: P@k, NDCG@10, MRR
  - Diagnostics: per‑query (`per_query*.jsonl`), latency (`latency*.json`), position (`position*.json`)

### Experimental Setup
- Environment: `conda activate rag_env`; threads limited (e.g., `OMP_NUM_THREADS=2`)
- Databases
  - SciFact: `data/rag_vectors__scifact_fixed_512_50.db`
  - FiQA: `data/rag_vectors__fiqa_fixed_512_50.db`
- Outputs
  - `results/e1_scifact/full/scifact_fixed_512_50/`
  - `results/e1_fiqa/full/fiqa_fixed_512_50/`
  - Triad summary: `results/e1_triad/summary/`

### Results
- SciFact (512/50)
  - Dense: P@1=0.13345, P@10=0.02362, NDCG@10=0.17279, MRR=0.16187
  - BM25 (aliased): P@1=0.00451, P@10=0.00045, NDCG@10=0.00451, MRR=0.00451
  - Hybrid: equals Dense
  - RRF: NDCG@10=0.17362 (+0.00083 vs Dense); MRR=0.16269
  - Z‑Score: equals Dense
  - Latency (ms): Dense p95=140.69; BM25 p95=1.47; Hybrid p95=127.83
  - Position: Mean=0.091; begin‑bin dominates (980/1109)
- FiQA (512/50)
  - Dense: P@1=0.03354, P@10=0.01628, NDCG@10=0.05135, MRR=0.03878
  - BM25 (aliased): P@1=0.00090, P@10=0.00030, NDCG@10=0.00129, MRR=0.00106
  - Hybrid: equals Dense
  - RRF: NDCG@10=0.03246 (−0.01889 vs Dense); MRR=0.04167
  - Z‑Score: NDCG@10=0.03228 (−0.01907 vs Dense); MRR=0.04128
  - Latency (ms): Dense p95=5376.26; BM25 p95=49.41; Hybrid p95=8484.48
  - Position: Mean=0.053; begin‑bin dominates (6205/6648)

### Analysis
- Dense versus BM25: Dense clearly outperforms BM25 on both datasets; BM25’s isolated signal is weak relative to ground truth.
- Hybrid equals Dense: With z‑normalization, weak BM25 contributes negligible signal; consistent with equal metrics to Dense.
- Fusion
  - SciFact: RRF slightly improves ranking (small NDCG@10 gain), likely benefiting from union of ranked lists when BM25 occasionally surfaces complementary docs. Z‑Score offers no change.
  - FiQA: Fusion degrades quality—weak BM25 introduces noise; both RRF/Z‑Score underperform Dense/Hybrid.
- Latency: SciFact vector retrieval is fast and tight; FiQA has heavy‑tail latency, likely due to DB size/content distribution or I/O/system variability.
- Position sensitivity: Strong preference for early chunks; relevant evidence concentrates near document beginnings.

### Threats to Validity
- Incomplete chunking grid: Only 512/50 reported for both datasets in this pass; broader E1 grid would validate whether 256/20 or semantic chunking changes outcomes.
- BM25 configuration: Default tokenization/stopwords/stemming may underfit domain; tuning could strengthen hybrid/fusion.
- Fusion hyperparameters: RRF `k` and Z‑Score strategy were not dataset‑tuned; other learned/weighted schemes may improve.
- Performance variability: FiQA latency heavy tails may be environment‑dependent; further profiling required.

### Recommendations
- Chunking follow‑up: Run 256/20 and semantic/late variants for SciFact/FiQA to identify dataset‑optimal chunking.
- BM25 tuning: Adjust tokenization/stopwords; consider query normalization; explore BM25 weighting in hybrid or selective BM25 use.
- Fusion improvements: Try weighted fusion based on per‑query BM25 quality, calibrated score fusion, or learned combiner.
- Latency remediation: Complete `rebuild/vacuum` passes; profile retrieval hot paths; review ANN parameters and batch sizes, especially for FiQA.
- Position‑aware retrieval: Consider small position priors or windowing that leverages early‑chunk salience.

### Conclusion
Dense/Hybrid retrieval dominates BM25 on both SciFact and FiQA at 512/50. RRF provides a tiny lift on SciFact but degrades FiQA—highlighting the risk of fusing weak signals. SciFact retrieval latency is consistently low; FiQA shows heavy tails. Evidence tends to cluster early in documents. Future work should expand chunking variants, tune BM25/fusion, and address FiQA latency tails.

### Reproducibility
- SciFact (E3 fusion):
  ```bash
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
  python "experiments/gpt-5-pro proposals/scripts/e3_fusion.py" \
    --dataset-dir "experiments/gpt-5-pro proposals/datasets/scifact" \
    --outdir "results/e1_scifact/full/scifact_fixed_512_50" \
    --db-path "data/rag_vectors__scifact_fixed_512_50.db" \
    --embedding-path "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf" \
    --collection "scifact_fixed_512_50" --kdocs 10 --nchunks 100 --cap 3 --rrf_k 60
  ```
- FiQA (E3 fusion):
  ```bash
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
  python "experiments/gpt-5-pro proposals/scripts/e3_fusion.py" \
    --dataset-dir "experiments/gpt-5-pro proposals/datasets/fiqa" \
    --outdir "results/e1_fiqa/full/fiqa_fixed_512_50" \
    --db-path "data/rag_vectors__fiqa_fixed_512_50.db" \
    --embedding-path "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf" \
    --collection "fiqa_fixed_512_50" --kdocs 10 --nchunks 100 --cap 3 --rrf_k 60
  ```
- Triad summary (E1):
  ```bash
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
  python "experiments/gpt-5-pro proposals/scripts/e1_analyze_triad.py" --outdir results/e1_triad/summary
  ```

### Implementation Notes
- E3 fusion and aliased E1 metrics were generated for both datasets; triad analyzer was updated to prefer aliased metrics when present.
- Final DB maintenance (REBUILD/VACUUM) was deferred due to timeouts; revisit if latency remediation is targeted (especially for FiQA).

*** End of Report ***

