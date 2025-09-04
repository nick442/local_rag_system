````markdown
# End‑to‑End Data & Evaluation Plan for Running **All 6 Experiments** (E1–E6)

> **Short answer to your question:**  
> Yes—you *can* run all six experiments on a **single corpus**. If you choose one, pick **TREC‑COVID** (long, technical documents) because it exercises chunking (E1) and context budgeting (E6) best, and still works for hybrid retrieval (E3), reranking (E4), and query expansion (E5).  
> **However**, for stronger external validity, keep the harness able to run a **triad of BEIR corpora** (SciFact, FiQA‑2018, TREC‑COVID). The steps below support **both modes** (single‑corpus and multi‑corpus).  
> The theoretical rationale for corpus/queries, chunking, hybrid fusion, reranking, and expansion choices is summarized in your lit review (esp. chunking vs late‑chunking; complementarity of sparse+dense; RRF and score normalization; position bias/optimal *k*; leakage‑aware expansion). :contentReference[oaicite:0]{index=0}

---

## 0) Repository Assumptions

- Your modular pipeline exposes components for **ingestion → chunking → indexing (BM25 + dense) → fusion → reranking → prompt building** with a metrics logger.
- Replace `CLI` with your repo’s command entry point or Python module, e.g. `python -m rag.cli` or `python cli.py`.
- Results land in `./results/…` as JSONL/CSV.

---

## 1) Choose Corpus Strategy

**Option A (single corpus; simplest):**  
- Use **BEIR/TREC‑COVID** (CORD‑19 subset). Long articles stress E1 (chunking) and E6 (context position and *k*).  
- You can run **all** E1–E6 here.

**Option B (recommended for robustness):**  
- Run on **three corpora**:
  - **SciFact** (claim verification; short abstracts),
  - **FiQA‑2018** (financial QA; noisier, user‑style queries),
  - **TREC‑COVID** (long biomedical docs).
- This gives coverage for short/long, lexical/semantic, and mixed/noisy queries.

> You can start with Option A to get initial results, then repeat the same pipeline on SciFact/FiQA with the same scripts/config matrix.

---

## 2) Acquisition (BEIR via `ir_datasets`)

**Tasks**
- [ ] Create `datasets/` with one subdir per corpus: `datasets/scifact`, `datasets/fiqa`, `datasets/trec_covid`.
- [ ] Add a tiny loader script `scripts/fetch_beir.py` that:
  - pulls **documents** (id, title, text, metadata),
  - **queries**,  
  - **qrels** (query–doc relevance),
  - writes them to `datasets/<name>/raw/` in **JSONL**:
    - `corpus.jsonl`: `{"doc_id": "...", "title": "...", "text": "...", "meta": {...}}`
    - `queries.jsonl`: `{"qid": "...", "text": "..."}`
    - `qrels.tsv`: standard TREC format (`qid 0 docid relevance`).

**Acceptance**
- [ ] Each dataset has `{corpus.jsonl, queries.jsonl, qrels.tsv}`.
- [ ] Doc counts match BEIR specs; random spot‑checks show intact text and useful metadata (e.g., journal, year).

---

## 3) Ingestion & Cleaning

**Goals**
- Normalize whitespace, drop boilerplate, de‑HTML if needed (TREC‑COVID PDFs often have artifacts).
- Preserve lightweight **metadata** fields (title, section headers if available).

**Tasks**
- [ ] `scripts/clean_corpus.py`:
  - Normalize unicode,
  - Remove HTML tags,
  - Merge `title + text` into a single `body` field (keep `title` separately for BM25 boost if supported),
  - Optional: filter trivially short docs (e.g., `< 30 tokens`).
- [ ] Write `datasets/<name>/clean/corpus.jsonl`.

**Acceptance**
- [ ] Token counts distribution (quick histogram) looks reasonable; no empty docs.
- [ ] Spot‑check 20 cleaned docs per corpus.

---

## 4) Chunking Pipelines (E1 foundation)

Implement **three** strategies with parameters:

1) **Fixed windows:** size ∈ {128, 256, 512}, overlap ∈ {0, 20, 50}  
2) **Semantic/heading split:** sentence/heading‑aware segmentation, then re‑pack up to target token limits  
3) **Late chunking:** encode full doc → pool token vectors into chunk vectors (compute‑heavier; done once per doc)

**Tasks**
- [ ] `scripts/chunk_corpus.py --strategy {fixed,semantic,late} --size N --overlap M`:  
  Outputs `datasets/<name>/chunks/<strategy>_<size>_<overlap>/chunks.jsonl` with:
  ```json
  {"chunk_id":"<docid>#<i>", "doc_id":"...", "text":"...", "offset": [start,end], "meta":{...}}
````

* [ ] For **late chunking**, add `--encoder e5-base-v2|bge-base` to precompute per‑doc token embeddings and pool into chunk vectors (store both **text chunk** + **vector** to avoid re‑encoding later).

**Acceptance**

* [ ] Size/overlap grids produce the expected number of chunks.
* [ ] For late chunking, confirm saved vectors have the correct dimension.

---

## 5) Indexing: Sparse (BM25) and Dense (E2 foundation)

### 5.1 Sparse: BM25 via Pyserini/Anserini

**Tasks**

* [ ] `scripts/index_bm25.py --dataset <name> --chunks <path>` builds a Lucene index in `indexes/<name>/bm25/<strategy>_<size>_<overlap>/`.
* [ ] Tokenization preset: English, stopwords on; optionally add **title boost** if present.

**Acceptance**

* [ ] Index builds without errors; term stats exist; sample searches return sensible docs.

### 5.2 Dense: Embeddings & Vector Store

**Encoders**

* **Primary:** `E5-base-v2` and `BGE-base-en-v1.5`
* **Precision variants:** fp32 and **fp16** (prefer fp16 on Mac; big memory win with minimal quality loss)

**Vector store**

* **Default:** `sqlite-vec` (portable, minimal deps)
* **Alternative:** `Faiss` (if you need HNSW/PQ or faster ANN)

**Tasks**

* [ ] `scripts/encode_chunks.py --encoder e5|bge --precision {fp32,fp16} --chunks <path> --out vectors/…`

  * Write to `vectors/<name>/<encoder>_<precision>/<strategy>_<size>_<overlap>/{.npy/.parquet}` with `chunk_id ↔ vector`.
* [ ] `scripts/build_dense_index.py --store sqlite-vec|faiss --in vectors/... --out indexes/<name>/dense/...`

  * For **Faiss**, support `--index hnsw --efSearch 64 --M 32` defaults.
  * For **sqlite-vec**, create a table `chunks(id TEXT PRIMARY KEY, meta JSON, vec BLOB)` and ensure correct **dimension**.

**Acceptance**

* [ ] Index sizes logged; memory estimate ≤ a few GB; test ANN lookups on 5 random queries.

---

## 6) Query Sets (Primary + Augmentations for E5/E6)

### 6.1 Primary (BEIR official)

* Use `queries.jsonl` and `qrels.tsv` as the **gold** evaluation set.

### 6.2 LLM‑based expansion (E5)

Run **three** augmentation tracks (all *scored against original qrels*):

1. **RM3** (classical PRF) over BM25.
2. **HyDE**: generate a 1–3 sentence “hypothetical answer/doc” per query (local LLM), then embed & retrieve with that text.
3. **Query2Doc**: generate a 2–4 sentence pseudo‑document per query (local LLM) and use it to expand BM25/dense queries.

**Leakage‑aware variant:** **CSQE** (corpus‑steered) — restrict the expansion terms to sentences extracted from already retrieved, **in‑corpus** passages (no OOD knowledge).

**Tasks**

* [ ] `scripts/make_query_sets.py --dataset <name> --mode {rm3,hyde,q2doc,csqe} --llm gemma-3-4b --max_tokens 256 --cache .cache/expansions`

  * Input: primary queries
  * Output: `datasets/<name>/queries_<mode>.jsonl` with `{"qid": "...", "text": "...", "expansion": "...", "meta": {...}}`.
* [ ] Add a `--perturb` option to produce **robustness** sets (synonym swaps, mild typos, entity masking) → `queries_perturbed.jsonl`.

**Acceptance**

* [ ] 100% coverage of qids; token length caps respected; cache hits logged; spot‑check expansions for topicality.

### 6.3 Context diagnostics (E6)

* Make a **needle‑in‑a‑haystack** mini‑set (10–20 items) by inserting a short fact sentence into long contexts and verifying retrieval + position sensitivity.
* Store in `datasets/<name>/diagnostics/nih.jsonl`.

---

## 7) Experiment Matrix (What to run)

Define a single YAML **matrix** so your agent can sweep settings *consistently across corpora*:

```yaml
# experiments/matrix.yaml
corpora: [trec_covid]   # or [scifact, fiqa, trec_covid]
retrievers:
  - bm25
  - dense: {encoder: e5-base-v2, precision: [fp32, fp16]}
  - dense: {encoder: bge-base-en-v1.5, precision: [fp32, fp16]}
fusion:
  - none
  - rrf
  - interpolate: {alpha: [0.2, 0.5, 0.8], norm: [z, minmax]}
  - dynamic_alpha: {features: [ner_density, oov_rate]}
rerank:
  - none
  - cross_encoder: {model: ms-marco-MiniLM-L6-v2, topN: [50]}
  - cross_encoder: {model: bge-reranker-v2-m3, topN: [50]}
chunking:
  - fixed: {size: [128,256,512], overlap: [0,20,50]}
  - semantic: {size: [256], overlap: [20]}
  - late: {encoder: [e5-base-v2,bge-base-en-v1.5], size: [256]}
queries:
  - primary
  - rm3
  - hyde
  - q2doc
  - csqe
  - perturbed
k_context: [2,4,6,8,12]     # E6
metrics: [ndcg@10, recall@1, recall@5, recall@20, mrr@10]
```

---

## 8) Orchestration Commands (per stage)

> **Adapt these to your CLI.** The idea is a reproducible pipeline the agent can run.

### 8.1 Build once per (corpus × chunking)

```bash
# Clean + chunk
CLI clean --dataset trec_covid
CLI chunk --dataset trec_covid --strategy fixed --size 256 --overlap 20
CLI chunk --dataset trec_covid --strategy semantic --size 256 --overlap 20
CLI chunk --dataset trec_covid --strategy late --encoder e5-base-v2 --size 256

# BM25 index per chunking variant
CLI index-bm25 --dataset trec_covid --chunks fixed_256_20
CLI index-bm25 --dataset trec_covid --chunks semantic_256_20
CLI index-bm25 --dataset trec_covid --chunks late_256

# Dense vectors + index (E5 fp16 & BGE fp16)
CLI encode --dataset trec_covid --chunks fixed_256_20 --encoder e5-base-v2 --precision fp16
CLI build-dense --dataset trec_covid --encoder e5-base-v2 --precision fp16 --store sqlite-vec

CLI encode --dataset trec_covid --chunks fixed_256_20 --encoder bge-base-en-v1.5 --precision fp16
CLI build-dense --dataset trec_covid --encoder bge-base-en-v1.5 --precision fp16 --store sqlite-vec
```

### 8.2 Query sets

```bash
# Primary uses official queries
# Expansions (HyDE/Q2Doc/CSQE) cached locally
CLI queries --dataset trec_covid --mode rm3
CLI queries --dataset trec_covid --mode hyde --llm gemma-3-4b --max_tokens 256
CLI queries --dataset trec_covid --mode q2doc --llm gemma-3-4b --max_tokens 256
CLI queries --dataset trec_covid --mode csqe --llm gemma-3-4b --max_tokens 256
CLI queries --dataset trec_covid --mode perturb
```

### 8.3 Run the experiment matrix

```bash
# Retrieval-only baselines
CLI eval --dataset trec_covid --queries primary \
  --retriever bm25 --metrics ndcg@10,recall@1,recall@5,recall@20,mrr@10 --out results/trec/bm25_primary.jsonl

CLI eval --dataset trec_covid --queries primary \
  --retriever dense --encoder e5-base-v2 --precision fp16 --metrics ... --out results/trec/e5_primary.jsonl

# Fusion: RRF & interpolation; Dynamic-α
CLI eval --dataset trec_covid --queries primary \
  --fusion rrf --candidates 100 --metrics ... --out results/trec/rrf_primary.jsonl

CLI eval --dataset trec_covid --queries primary \
  --fusion interpolate --alpha 0.5 --norm z --candidates 100 --out results/trec/interp05z_primary.jsonl

CLI eval --dataset trec_covid --queries primary \
  --fusion dynamic-alpha --features ner_density,oov_rate --candidates 100 --out results/trec/dalpha_primary.jsonl

# Reranking
CLI rerank --in results/trec/rrf_primary.jsonl \
  --model ms-marco-MiniLM-L6-v2 --topN 50 --out results/trec/rrf_minilm.jsonl

# Query expansion (repeat with rm3, hyde, q2doc, csqe)
CLI eval --dataset trec_covid --queries hyde --fusion rrf --candidates 100 --out results/trec/rrf_hyde.jsonl

# Context budgeting (E6) – vary k and order
CLI eval --dataset trec_covid --queries primary --fusion rrf --rerank ms-marco-MiniLM-L6-v2 \
  --k_context 2,4,6,8,12 --order {relevance,random} --out results/trec/context_sweep.jsonl
```

---

## 9) Evaluation & Statistics

**Tasks**

* [ ] `analysis/aggregate.py`: compute **nDCG\@10, Recall\@k, MRR\@10**, and latency summaries for each run.
* [ ] `analysis/significance.py`: **paired randomization** and **paired t‑test** across *the same queries*; output **p‑values** and **95% CIs**.
* [ ] `analysis/plots.py`:

  * Pareto plots (**nDCG\@10 vs ms/query vs MB**),
  * ΔnDCG\@10 bar charts (fusion, reranking, expansion),
  * Context‑*k* curves with relevance‑ordered vs random.

**Acceptance**

* [ ] CSV/JSON summaries per (dataset × method) with metrics + 95% CIs.
* [ ] Plots saved to `figures/`.

---

## 10) Resource & Run‑Time Tips (Mac mini, 16 GB)

* Prefer **fp16** embeddings; 384–768 dims.
* Limit reranker to **topN ≤ 50**; MiniLM‑L6 is a good default; BGE‑reranker if multilingual.
* Use batch sizes 8–16 for reranking on CPU; cache expansions.
* If memory tight: use **sqlite‑vec** over Faiss, or Faiss with HNSW/PQ and `efSearch` lowered (accuracy–speed trade‑off).

---

## 11) Deliverables & Folder Layout

```
datasets/
  trec_covid|scifact|fiqa/
    raw/{corpus.jsonl,queries.jsonl,qrels.tsv}
    clean/corpus.jsonl
    chunks/<strategy>_<size>_<overlap>/chunks.jsonl
    diagnostics/nih.jsonl
indexes/
  <name>/bm25/<chunking>/
  <name>/dense/<encoder>_<precision>/<chunking>/
vectors/
  <name>/<encoder>_<precision>/<chunking>/
results/
  <name>/*.jsonl
figures/
experiments/matrix.yaml
scripts/*.py
analysis/*.py
```

---

## 12) Minimal Acceptance Checklist (per corpus)

* [ ] Acquisition, cleaning, and **three** chunking strategies materialized.
* [ ] BM25 + dense indexes built (E5 fp16, BGE fp16 at minimum).
* [ ] Query sets: primary + {RM3, HyDE, Q2Doc, CSQE, perturbed}.
* [ ] Baselines run: bm25, dense (e5/bge), hybrid (RRF + interpolate), **then** reranking (MiniLM).
* [ ] Context‑*k* sweep with two orderings.
* [ ] Aggregates + significance + plots generated.
* [ ] Final CSV/JSON ready for the thesis Results chapter.

---

## 13) FAQ

* **Can we really do everything on one corpus?**
  Yes. Use **TREC‑COVID**. For completeness, re‑run on **SciFact** (short/precise) and **FiQA** (noisy/mixed) when time permits.

* **Which chunking should be the default before E1 ends?**
  `fixed_256_20` is a robust default; semantic and late‑chunking are compared in E1.

* **Which embedding default?**
  Start with **E5‑base‑v2 (fp16)**; compare with **BGE‑base‑en‑v1.5 (fp16)** in E2.

* **Which fusion default?**
  **RRF**—strong, training‑free; add `$z$`‑normalized interpolation and dynamic‑α in E3.

* **Which reranker default?**
  **MiniLM‑L6‑v2** for speed; try **BGE‑reranker‑v2‑m3** for robustness.

* **Expansion default?**
  Prefer **CSQE** (corpus‑steered) to reduce leakage risk; HyDE/Query2Doc as ablations.

---

### Notes on Theory Alignment

This plan operationalizes your lit review’s core insights:

* **Chunking** (fixed vs semantic vs **late**), **position bias** and optimal **k**;
* **Sparse–dense complementarity**, **RRF** and score normalization, plus a **dynamic‑α** heuristic;
* **Compact rerankers** on CPU;
* **LLM‑based expansions** with leakage‑aware **CSQE**.
  See your PDF, pp. 2–9, 11–13, for the theoretical grounding and formulas that motivated these steps.&#x20;

```
::contentReference[oaicite:2]{index=2}
```
