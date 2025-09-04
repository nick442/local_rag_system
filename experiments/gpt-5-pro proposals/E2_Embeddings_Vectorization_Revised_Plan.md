# E2 — Embedding & Vectorization Execution Plan (Revised)

**Goal**  
Quantify how **embedding backbones**, **vector precision/size**, and **indexing choices** affect **retrieval quality** and **cost** on a 16 GB local RAG stack. Chunking, scoring, tokenizer, PRAGMAs, and evaluation remain **constant** (best fixed setting from E1).

**Key Decisions**  
- Compare **E5‑base‑v2** vs **BGE‑base‑en‑v1.5** (optionally MiniLM‑L6‑v2 as a speed lower bound).  
- Vector dtype/size: **fp32 vs fp16** (and int8 if supported); dim **384 vs 768** (if model supports).  
- Index backend constant (sqlite‑vec **or** Faiss); ANN params fixed.  
- **Doc‑aware ranking** and BEIR metrics (NDCG@10 primary) with significance tests.  
- Minimal **BM25 & Hybrid** control runs to show effects aren’t dense‑only artifacts.

---

## 0) Design Principles & Controls
- **Fixed from E1:** chunking = **best fixed** (e.g., 256/20), tokenizer `cl100k_base`, PRAGMAs, candidate **Nchunks=100**, **per‑doc cap C=3**, doc‑aware ranking, evaluation protocol.  
- **Variable:** embedding model, vector dimensionality/precision.  
- Keep **retrieval kdocs=10** for the primary grid; run **k‑sensitivity** on the top model later.

---

## 1) Corpora & Layout
Use the E1 triad: **TREC‑COVID**, **SciFact**, **FiQA‑2018** (BEIR via `ir_datasets`). Local folder layout identical to E1.

Validation: doc‑ID parity (qrels ⊆ ingested docs), counts match, parsers OK.

---

## 2) Models & Index Variants
- **Embedders:**  
  - `intfloat/e5-base-v2` (768‑dim)  
  - `BAAI/bge-base-en-v1.5` (768‑dim)  
  - *(Optional baseline)* `sentence-transformers/all-MiniLM-L6-v2` (384‑dim)
- **Vector dtype:** fp32, **fp16** (preferred if backend supports), *(optional)* int8 PQ in Faiss.  
- **Backends:** choose **one** for the main grid (sqlite‑vec for simplicity **or** Faiss for PQ/HNSW). Keep it constant.

**DB strategy**  
- To avoid schema churn, create **separate collections/DBs per dimension** (384 vs 768).  
- Log vector dtype & ANN parameters in the ledger.

---

## 3) Retrieval & Doc‑Aware Ranking
- **Dense** (primary): selected embedder & dtype; ANN params fixed; candidate **Nchunks=100**; **C=3**; doc‑aware ranked **Kdocs=10**.  
- **Controls (minimal):** BM25 and **Hybrid(RRF)** on **one dataset** (TREC‑COVID) using the **best** embedder/dtype after the dense grid.

---

## 4) Metrics & Performance Ledger
**Retrieval (doc‑level):** NDCG@10 (primary), MAP, MRR@10, Recall@{1,5,10,20}.  
**Cost:** embedding throughput (tokens/s), **embed time**, index build time, **DB size**, **RSS peak**, retrieval latency p50/p95, **#chunks**, **dup@N**.

Emit: `results/e2_<dataset>/<variant>/{metrics.json,per_query.jsonl,ledger.jsonl}`.

---

## 5) Smoke Test
- Dataset: **TREC‑COVID**, variants: **E5‑fp32**, **BGE‑fp32**.  
- Steps: ingest with E1 chunking; build dense index; run 50‑query stratified set; verify metrics non‑zero, ledger captured.  
- Optional: BM25 & Hybrid(RRF) on the **better** of the two to sanity‑check interactions.

---

## 6) Full Grid (Per Dataset)
**Dense grid:**  
- Models: {E5‑base‑v2, BGE‑base‑en‑v1.5}  
- Dtypes: {fp32, **fp16**} → 4 variants (8 if you also include MiniLM‑384).

**Procedure:** ingest/build per variant; run full queries with doc‑aware ranking; compute metrics; ledger logging.

**Statistics:** paired t‑test + bootstrap vs **E5‑fp32** (reference). Report mean ±95% CI and p‑values.

**K‑sensitivity:** On the **best two** variants, re‑run with **Kdocs ∈ {5,10,20}** (TREC‑COVID only).

---

## 7) Scale & Memory Guidance
- 768‑dim fp32 vectors are ~3 KB each; fp16 halves memory. Keep chunks ≤ ~200k to stay within 16 GB with overhead.  
- If using Faiss PQ (optional), note accuracy trade‑offs; keep code paths identical across variants.

---

## 8) Operational Details & Failure Modes
- **Fallback dtype:** If fp16 unsupported, record and continue with fp32; mark variant accordingly.  
- **Tokenizer drift:** Ensure the same tokenizer used for E1 chunking is applied during embedding (token counts for throughput).  
- **RSS spikes:** Reduce batch size for embedding (e.g., 16) and ANN search parameters.

---

## 9) Work Items
- Add model registry/CLI flags: `--embedder MODEL --embed-dtype {fp32,fp16}`.  
- Ensure backend can load/store fp16; if not, cast on ingest and annotate in ledger.  
- Eval runner `scripts/e2_eval.py` to iterate variants and produce tables/plots (Pareto NDCG vs latency vs memory).

---

## 10) Checklists
- [ ] E1 chunking fixed, PRAGMAs verified  
- [ ] E5/BGE models downloaded and smoke tested  
- [ ] Dense grid completed per dataset; stats/plots generated  
- [ ] Optional controls (BM25/Hybrid) done on TREC‑COVID

---

## 11) Command Snippets
```bash
# Ingest with E1-best chunking, using E5 fp16
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_e5_fp16 --embedder e5-base-v2 --embed-dtype fp16

# Evaluate (dense)
python scripts/e2_eval.py --dataset trec_covid \
  --collections trec_covid_e5_fp16 trec_covid_bge_fp16 \
  --retrieval dense --kdocs 10 --nchunks 100 --per_doc_cap 3

# Optional: controls on best variant
python scripts/e2_eval.py --dataset trec_covid --collections trec_covid_e5_fp16 --retrieval bm25 --kdocs 10
python scripts/e2_eval.py --dataset trec_covid --collections trec_covid_e5_fp16 --retrieval hybrid_rrf --kdocs 10
```
