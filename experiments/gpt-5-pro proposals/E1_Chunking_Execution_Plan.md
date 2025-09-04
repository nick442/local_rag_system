# E1 — Chunking Execution Plan (Revised)

**Goal**  
Quantify how **chunking strategy** and **parameters** (size, overlap, semantics, late‑chunking proxy) impact **retrieval quality** and **cost** in a local RAG stack. We **hold models, vector backend, PRAGMAs, tokenizer, and scoring constant** and vary **chunk boundaries only**. Results are reported at the **document level** with **paired statistical tests**.

**Key Improvements over the original plan**  
- **Doc‑aware ranking**: ensure fair, comparable **K docs** across variants (avoid long‑doc & multi‑chunk bias).  
- **Sparse/Hybrid controls**: add minimal **BM25** and **Hybrid (RRF / α‑mix)** passes to show effects generalize beyond dense retrieval.  
- **Cost ledger**: log **index size, embed/build time, RSS, latency p50/p95, chunk counts, dup@N** to plot **NDCG vs latency vs memory**.  
- **Concrete semantic and late‑chunking definitions** sized for **16 GB**.  
- **k‑sensitivity & ordering**: sweep **Kdocs ∈ {5,10,20}**, and enforce **relevance‑descending** doc ordering to mitigate “lost‑in‑the‑middle.”  
- **Stratified 50‑query smoke set** with fixed seed for reproducibility.

---

## 0) Design Principles & Controls

- **Constant factors:** embedder = MiniLM‑L6‑v2 (384‑dim), tokenizer = `cl100k_base`, vector store = sqlite‑vec (or Faiss if already selected), PRAGMAs (WAL, page_size, cache_size), scoring normalization, ANN parameters, candidate **Nchunks** and **per‑doc cap**.  
- **Variable:** chunk boundaries only (size, overlap, backend).  
- **Evaluation unit:** **documents** (fold chunks→docs; rank unique docs; score vs qrels).  
- **Significance:** paired tests (t‑test + bootstrap) over queries; report mean ± **95% CI** and p‑values; note multiple‑testing where applicable.

---

## 1) Corpus Acquisition & Validation

**Primary:** **TREC‑COVID** (BEIR split via `ir_datasets`) → then **SciFact**, **FiQA‑2018** after pipeline validation.

**Local layout (not committed):**
```
datasets/trec_covid/docs/*.txt      # one file per doc (title + body)
datasets/trec_covid/queries.jsonl   # {"qid": str, "text": str}
datasets/trec_covid/qrels.tsv       # qid \t 0 \t docid \t relevance
```
**Export options**  
- _Option A (scripted)_: `scripts/fetch_beir.py` to dump to the above layout.  
- _Option B (once‑off)_: small Python utility using `ir_datasets` to write docs/queries/qrels.

**Validation & acceptance**  
- **Doc ID parity**: assert `qrels.docids ⊆ docs_ids` (fail‑fast if not).  
- Spot‑check ~20 docs for content integrity.  
- Verify `queries.jsonl` and `qrels.tsv` parse; counts match BEIR specs.

---

## 2) Ingestion & Collections (per chunking variant)

We **materialize each chunking variant** into a distinct `collection_id`. Retrieval already scopes by collection.

**CLI enhancements (PR‑E1‑CLI):**
- Add to `main.py ingest directory`:  
  `--chunk-size INT`, `--chunk-overlap INT`, `--chunk-backend {fixed,semantic}`  
- Thread through to `DocumentIngestionService`. Keep tokenizer fixed.  
- Add **content cleanup** (HTML strip, whitespace normalize) before chunking.

**SQLite PRAGMAs (constant across variants):**  
`journal_mode=WAL`, `synchronous=NORMAL`, `page_size=4096(or 8192)`, `cache_size=-200000`, `mmap_size=268435456`. Capture in logs.

**Embedding dimension**  
- Keep 384‑dim to avoid reindexing confounds. If switching to 768‑dim in later experiments, use a **new DB**.

**Acceptance**  
- Ingestion completes; **#docs** equals BEIR doc count; non‑zero **#chunks**; logs capture ingest/embedding times and DB size.

---

## 3) Chunking Variants (precisely defined)

### 3.1 Fixed windows (grid)
- **Sizes:** {128, 256, 512} tokens  
- **Overlaps:** {0, 20, 50} tokens  
- Sliding window; boundaries by tokenizer; store `chunk_index`, `token_span` metadata.

### 3.2 Semantic splitter (deterministic, lightweight)
- **Step 1:** Split on **section headings** (Markdown/HTML titles when available).  
- **Step 2:** Within a section, split to **sentences** (spaCy).  
- **Step 3:** **Greedy pack** sentences up to target size (e.g., 256); **no overlap**.  
- **Policy:** include the **heading text** in the first chunk of each section (as metadata + prefix tokens).  
- **Rationale:** preserves topical coherence; minimizes redundancy; cheap to run.

### 3.3 Late‑chunking proxy (practical for 16 GB)
- **Doc retrieval stage:** compute **doc vector** as mean of **sentence embeddings** (MiniLM). ANN over docs → **top‑M docs** (M=20–50).  
- **Slice‑and‑score stage:** within these docs, slide a fixed window (256/50), score each chunk by `cosine(query, chunk)`.  
- **Selection:** form **candidate chunk pool** only from these M docs; then apply **doc‑aware ranking** (Section 4).  
- **Rationale:** emulate “encode first, split later” without long‑context encoders.

---

## 4) Retrieval & **Doc‑Aware Ranking** Protocol

### 4.1 Retrieval modes for E1
- **Dense (baseline):** MiniLM vectors via sqlite‑vec (constant ANN params).  
- **BM25 (control):** Pyserini/FTS index; defaults.  
- **Hybrid (control):** (a) **α‑mix** of z‑scored dense + BM25 with α ∈ {0.5}, (b) **RRF** (k=60).  
> Controls are run on **two fixed variants** to demonstrate chunking effects generalize beyond dense.

### 4.2 Candidate formation & **doc‑aware** ranking (mandatory)
Given a query:
1. Retrieve **top‑Nchunks** (e.g., N=100) by the selected mode.  
2. **Per‑doc cap C**: keep at most **C=3** chunks per doc in the pool (mitigates long‑doc dominance).  
   _Alternative_: apply **MMR** on chunks for diversity before folding.  
3. **Fold to docs**: for each `doc_id`, keep the **best chunk** (by score, tie‑break by min chunk rank).  
4. **Rank unique docs** by best‑chunk score (and best rank as tie‑break).  
5. Evaluate **top‑Kdocs** (K=10 primary; see K‑sensitivity).

Log **dup@N** = fraction of top‑Nchunks that belong to already‑seen docs.

---

## 5) Query Sets

- **Primary:** official BEIR `queries.jsonl` + `qrels.tsv` for TREC‑COVID.  
- **Stratified smoke set (50 queries):** fixed seed; stratify by **query length** (short/long) and **entity count** (0/1/2+). Persist file to `datasets/trec_covid/queries_small.jsonl`.

> E1 focuses on retrieval; no LLM expansions used.

---

## 6) Metrics & **Performance Ledger**

### 6.1 Retrieval metrics (doc‑level)
- **Primary:** **NDCG@10** (graded qrels).  
- **Also:** **MAP**, **MRR@10**, **Recall@{1,5,10,20}**.  
- Compute **per query**, export per‑query JSONL for stats.

### 6.2 Cost & system metrics (per variant)
- **#docs**, **#chunks**, **avg tokens/chunk**, **overlap tokens**  
- **Embedding throughput** (tokens/sec), **embed time**, **index build time**  
- **DB size (MB)**, **RSS peak (MB)** during retrieval  
- **Retrieval latency** p50/p95 (ms)  
- **dup@N** at N=50 or 100

All emitted to `results/e1_trec_covid/<variant>/ledger.jsonl`.

---

## 7) Smoke Test (small sweep)

**Configs:** `fixed_256_20`, `fixed_512_50` (and existing `fixed_512_128` if helpful)  
**Retrieval:** Dense (baseline), **plus BM25 & Hybrid** on `fixed_256_20` **only** (control).  
**Kdocs:** 10; **Nchunks:** 100; **C (per‑doc cap):** 3.

**Steps**
1. Ingest each variant collection.  
2. Run on 50‑query stratified set; produce doc‑aware rankings; compute metrics & ledger.  
3. Sanity checks: non‑zero metrics, topical relevance on spot‑check, stable latency, memory within limits.  
4. Log PRAGMAs and ANN params in each run.

**Artifacts**
```
results/e1_trec_covid/test/metrics_<variant>.json
results/e1_trec_covid/test/per_query_<variant>.jsonl
results/e1_trec_covid/test/ledger_<variant>.jsonl
```
**Acceptance**: All runs complete; metrics/ledger exist; control (BM25/Hybrid) runs complete on one variant.

---

## 8) Full Grid Evaluation (TREC‑COVID) + K‑Sensitivity

**Variants:** Fixed grid = sizes {128,256,512} × overlaps {0,20,50} ⇒ **9 variants**.  
**Retrieval:** Dense only (primary grid).  
**K‑sensitivity:** For **best two** fixed configs by NDCG@10, re‑evaluate with **Kdocs ∈ {5,10,20}**.  
**Doc pool:** Nchunks=100 (constant); C=3.

**Procedure**
1. Ingest all variants (batch).  
2. For each variant, run full query set; **doc‑aware** ranking; compute metrics; log ledger.  
3. **Statistics:** Paired **t‑test** and **bootstrap** vs. reference `fixed_256_20`; report mean ± 95% CI and p‑values. Note exploratory vs. confirmatory; mention multiple‑testing.

**Outputs**
```
results/e1_trec_covid/full/<variant>/metrics.json
results/e1_trec_covid/full/<variant>/per_query.jsonl
results/e1_trec_covid/full/<variant>/ledger.jsonl
reports/e1_trec_covid/summary.csv
reports/e1_trec_covid/plots/* (NDCG vs latency/memory; K-sensitivity)
```

---

## 9) Scale‑Out (SciFact, FiQA‑2018)

Repeat Sections **1–8** on **SciFact** and **FiQA‑2018** with the same grid (fixed only is acceptable if time‑boxed). Keep PRAGMAs and ANN params constant. Produce a **triad summary** table.

---

## 10) Operational Details & Failure Modes

**Environment**  
```
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rag_env
python ...
```

**Performance tips**  
- Batch size for embeddings: 32; retrieval Kdocs ≤ 10 in primary grid; Nchunks=100; C=3.  
- If memory tight: ingest in slices (10k docs) per collection; VACUUM after ingest.

**Failures & mitigations**  
- **sqlite‑vec missing** → fallback to manual dot‑product: record in ledger; expect slower latency, still valid.  
- **Embedding dim mismatch** → new DB or re‑ingest.  
- **Parsing errors** → log and skip; keep counts.

---

## 11) Work Items (PRs & Scripts)

- **PR‑E1‑CLI:** `--chunk-size`, `--chunk-overlap`, `--chunk-backend {fixed,semantic}`; add cleanup step.  
- **PR‑E1‑SEM:** implement **semantic splitter** (headings + sentences; greedy pack, no overlap).  
- **PR‑E1‑LATE:** implement **late‑chunking proxy** (doc ANN → slice‑and‑score).  
- **PR‑E1‑RANK:** add **doc‑aware ranking** utility (fold chunks→docs with per‑doc cap/MMR).  
- **Script‑E1‑Eval:** evaluation runner: loads queries/qrels, executes retrieval per collection, doc‑aware ranking, metrics + ledger output.  
- **Script‑BEIR‑Dump:** export BEIR datasets to local layout with ID normalization/validation.  
- **Script‑Stats:** paired t‑test + bootstrap over per‑query deltas; emit CI tables.

---

## 12) Checklists

**Smoke test (TREC‑COVID)**  
- [ ] Docs dumped; qrels doc IDs ⊆ ingested doc IDs  
- [ ] Ingested `trec_covid_fixed_256_20`, `trec_covid_fixed_512_50`  
- [ ] 50‑query stratified run for dense; BM25 & Hybrid run on `fixed_256_20`  
- [ ] Metrics & ledger JSON/JSONL written; PRAGMAs recorded

**Full run (TREC‑COVID)**  
- [ ] All 9 fixed variants ingested as separate collections  
- [ ] Full query set evaluated per variant with **doc‑aware ranking**  
- [ ] K‑sensitivity completed for top‑2 variants  
- [ ] Summary tables, plots, and significance tests vs `fixed_256_20`

**Scale‑out**  
- [ ] SciFact and FiQA‑2018 acquired and validated  
- [ ] Repeat pipeline; produce triad summary

---

## 13) Command Snippets

**Status & collections**
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
python main.py status
python main.py collection list
```

**Ingest (fixed variants)**
```bash
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_fixed_256_20 --chunk-size 256 --chunk-overlap 20 --chunk-backend fixed

python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_fixed_512_50 --chunk-size 512 --chunk-overlap 50 --chunk-backend fixed
```

**Semantic (if PR‑E1‑SEM ready)**
```bash
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_semantic_256_0 --chunk-size 256 --chunk-overlap 0 --chunk-backend semantic
```

**Single‑query sanity check**
```bash
python main.py query "What treatments were studied for COVID-19?" \
  --collection trec_covid_fixed_256_20 --k 10 --metrics
```

**Analytics overview**
```bash
python main.py analytics stats --collection trec_covid_fixed_256_20
```

**Evaluation (script)**  
```bash
python scripts/e1_eval.py \
  --dataset trec_covid \
  --collections trec_covid_fixed_256_20 trec_covid_fixed_512_50 \
  --retrieval dense \
  --kdocs 10 --nchunks 100 --per_doc_cap 3
# Optional control:
python scripts/e1_eval.py --dataset trec_covid --collections trec_covid_fixed_256_20 --retrieval bm25 --kdocs 10
python scripts/e1_eval.py --dataset trec_covid --collections trec_covid_fixed_256_20 --retrieval hybrid_rrf --kdocs 10
```

---

## 14) Deliverables

- Per‑variant: `metrics.json`, `per_query.jsonl`, `ledger.jsonl`  
- Corpus triad summary: `reports/*/summary.csv`, plots (NDCG@10 vs latency/memory; K‑sensitivity)  
- Stats report: CI tables, p‑values, and interpretation
