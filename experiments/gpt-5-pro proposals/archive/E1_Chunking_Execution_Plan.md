# Experiment 1 (E1) — Chunking Execution Plan

Goal: Measure how chunking strategy and parameters impact retrieval quality and performance in our local RAG stack. We hold models, indexing backends, and evaluation constant, and vary chunk boundaries only.

Scope: Fixed windows (sizes/overlaps), plus semantic and late-chunking variants. Start on a single corpus (TREC‑COVID) for speed; scale to the BEIR triad (SciFact, FiQA‑2018, TREC‑COVID) for robustness once the pipeline is validated.

---

## 1) Corpus Acquisition

Primary: TREC‑COVID (BEIR split via `ir_datasets`). Optional follow‑ups: SciFact and FiQA‑2018.

- Target layout (local, not committed):
  - `datasets/trec_covid/docs/*.txt` — one text file per document (title + body)
  - `datasets/trec_covid/queries.jsonl` — `{ "qid": str, "text": str }`
  - `datasets/trec_covid/qrels.tsv` — `qid 0 docid relevance`

Acquisition steps (recommended):
- Option A (scripted): add a lightweight `scripts/fetch_beir.py` later to export BEIR to the above layout.
- Option B (manual): run a short Python one‑off (outside this plan) to dump docs to `datasets/trec_covid/docs/` and save queries/qrels.

Acceptance:
- Files present with doc count matching BEIR specs; spot‑check ~20 docs; `queries.jsonl` and `qrels.tsv` exist and parse.

---

## 2) Ingestion & Collections (per chunking variant)

Strategy: isolate chunking by materializing each variant into a distinct `collection_id`. Retrieval already supports `collection_id`.

Current CLI defaults chunking to `size=512`, `overlap=128`. For E1 we plan multiple variants; because `main.py ingest` does not expose chunk params yet, we will:
- Smoke‑test with the default first (to validate the pipeline).
- Then add a small CLI enhancement (separate PR) to pass `--chunk-size` and `--chunk-overlap` through to `DocumentIngestionService`.

Commands (smoke test now; fixed defaults):

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_fixed_512_128 --dry-run

source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_fixed_512_128

source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py collection list

source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py analytics stats --collection trec_covid_fixed_512_128
```

Planned CLI enhancement (separate PR):
- Add `--chunk-size` and `--chunk-overlap` to `ingest directory` and thread into `DocumentIngestionService`.

Post‑enhancement variant ingests (examples):

```bash
# Fixed 256/20
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_fixed_256_20 --chunk-size 256 --chunk-overlap 20

# Fixed 512/50
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_fixed_512_50 --chunk-size 512 --chunk-overlap 50

# Semantic 256/20 (requires semantic splitter integration; treat as future variant)
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_semantic_256_20 --chunk-size 256 --chunk-overlap 20
```

Embedding dimension note:
- The DB is initialized with a fixed embedding dimension (defaults to 384 via MiniLM). Keep the same encoder across chunking runs to avoid dimension mismatch. If switching to 768‑dim encoders (e.g., E5/BGE), create a separate DB or reindex from scratch.

Acceptance:
- Each variant collection has comparable document counts and reasonable chunk counts; no empty chunks; ingestion completes without errors.

---

## 3) Query Sets for E1

Primary: official BEIR `queries.jsonl` + `qrels.tsv` for TREC‑COVID. E1 focuses on retrieval; no LLM expansions.

- For quick tests, select a 50‑query subset (`datasets/trec_covid/queries_small.jsonl`).
- For full runs, use all queries.

Mapping chunks → docs for evaluation:
- Retrieve chunk IDs but score at the document level by mapping `chunk_id → doc_id`, then deduplicate per doc before metric computation (NDCG@10, Recall@k, MRR@10).

---

## 4) Test Run (small configuration sweep)

Purpose: Validate end‑to‑end ingestion → retrieval → metrics on a small subset, measure runtime, and sanity‑check outcomes before scaling.

Configurations:
- Variants: `fixed_256_20`, `fixed_512_50` (plus the default `fixed_512_128` if helpful)
- Retrieval: hold constant to `vector` (MiniLM 384‑dim); k = 10

Steps:
1) Ingest the two collections (see Section 2).
2) Load 50 queries from `queries_small.jsonl`.
3) For each collection, run retrieval via `RAGPipeline` (vector method) and capture top‑k chunk IDs; fold to doc IDs.
4) Compute metrics using `src/evaluation_metrics.py` (NDCG@10 primary; Recall@1/5/20; MRR@10). Save JSON summary.

Sanity checks:
- Metric values are non‑zero; tops docs look topically correct on spot‑check.
- Retrieval time/query is stable; memory use within local limits.

Helpful manual smoke check:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py query "What treatments were studied for COVID-19?" \
  --collection trec_covid_fixed_256_20 --k 5 --metrics
```

Artifacts:
- `results/e1_trec_covid/test/metrics_fixed_256_20.json`
- `results/e1_trec_covid/test/metrics_fixed_512_50.json`

Acceptance:
- Both runs complete; metrics JSON exists; runtime estimates recorded.

---

## 5) Big Run (full evaluation)

Purpose: Produce statistically robust, thesis‑ready results for E1.

Corpora:
- Start: TREC‑COVID.
- Scale: add SciFact and FiQA‑2018 once TREC‑COVID completes.

Variants to run:
- Fixed window grid: sizes ∈ {128, 256, 512}, overlaps ∈ {0, 20, 50} ⇒ 9 variants
- Optional: Semantic 256/20
- Optional: Late‑chunking 256 (requires pooled token embeddings; a separate ingestion path)

Retrieval settings:
- Retrieval method: `vector` (MiniLM 384‑dim)
- k ∈ {5, 10} (primary: 10)
- Keep LLM out of the loop for scoring (retrieval‑only metrics)

Execution outline:
1) Ingest all variant collections (batch with the enhanced CLI flags).
2) Run evaluation script that:
   - Iterates variants, runs retrieval for all queries, maps chunk → doc, computes metrics.
   - Logs per‑query scores for statistical tests (paired t‑test/bootstrapping across queries).
3) Aggregate results and produce summary tables/plots.

Result storage:
- `results/e1_trec_covid/full/<variant>/metrics.json`
- `results/e1_trec_covid/full/<variant>/per_query.jsonl`
- `reports/e1_trec_covid/summary.csv` and `reports/e1_trec_covid/plots/`

Statistical analysis:
- Paired significance against the reference `fixed_256_20`.
- Report mean ± 95% CI for NDCG@10 and Recall@k.

Acceptance:
- All variant runs complete; summaries and per‑query outputs exist; significance tables generated; plots render.

---

## 6) Operational Details

Environment:
- Always run via: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python ...`

Performance tips:
- Keep MiniLM (384‑dim) for E1 to avoid reindexing; batch size 32; k ≤ 10.
- Use collections per variant to isolate indexes.
- If memory becomes tight, process the corpus in slices (per 10k docs) per collection.

Failure modes & mitigations:
- sqlite‑vec not loaded: fallback path in `vector_database.py` uses manual similarity; performance drops but results remain valid.
- Embedding dimension mismatch: new DB or re‑ingest with consistent encoder.
- PDF parsing errors: ingestion logs and skips problematic pages; overall run continues.

---

## 7) Work Items & Ownership

- PR‑E1‑CLI: add `--chunk-size` and `--chunk-overlap` to `main.py ingest directory` and thread to `DocumentIngestionService`.
- (Optional) PR‑E1‑SEM: add semantic splitter (heading/sentence aware) and wire as a `--chunk-backend semantic` option.
- (Optional) PR‑E1‑LATE: add a late‑chunking pipeline that pools token embeddings to chunk vectors at retrieval time; store vectors to avoid re‑encoding.
- Script‑E1‑Eval: small evaluation script that
  - loads `queries.jsonl` and `qrels.tsv`,
  - runs retrieval via `RAGPipeline` for a given `collection_id`,
  - folds chunk→doc, computes metrics with `src/evaluation_metrics.py`,
  - writes `metrics.json` and `per_query.jsonl`.

---

## 8) Checklists

Smoke test (TREC‑COVID):
- [ ] Docs dumped to `datasets/trec_covid/docs/`
- [ ] Ingested `trec_covid_fixed_512_128`
- [ ] 50‑query small run completes; metrics JSON written

Full run (TREC‑COVID):
- [ ] All fixed window variants ingested as separate collections
- [ ] Full query set evaluated for each variant
- [ ] Summary tables and plots generated
- [ ] Significance tests vs `fixed_256_20` reported

Scale‑out:
- [ ] SciFact and FiQA‑2018 acquired and ingested
- [ ] Repeat the above pipeline; collect triad summary

---

## 9) Command Snippets (for reference)

Status and basic checks:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python main.py status
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python main.py collection list
```

Single query sanity check per collection:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py query "What interventions reduced transmission?" \
  --collection trec_covid_fixed_512_128 --k 5 --metrics
```

Analytics overview:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py analytics stats --collection trec_covid_fixed_512_128
```

---

Deliverables:
- Clean ingestion logs; per‑variant metrics JSON; per‑query outputs; summary CSV; significance report; plots. These feed directly into the E1 section of the Results chapter.

