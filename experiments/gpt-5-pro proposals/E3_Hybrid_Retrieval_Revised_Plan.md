# E3 — Hybrid Retrieval Execution Plan (Revised)

**Goal**  
Quantify gains from **combining sparse (BM25)** and **dense** retrieval under local constraints, and compare **fusion strategies**: **weighted α‑mix**, **Reciprocal Rank Fusion (RRF)**, and **Dynamic‑α (DAT)** gating. Report accuracy‑vs‑latency trade‑offs with doc‑aware ranking.

**Key Decisions**  
- Sparse: **BM25** via Pyserini (Lucene) or SQLite FTS5 (choose one and keep constant).  
- Dense: best embedder/dtype from **E2**.  
- Fusion: **z‑score α‑mix** (α∈{0.2, 0.5, 0.8}), **RRF(k=60)**, and **DAT** (heuristic/LLM‑gated α per query).  
- Candidate pool **Nchunks=100**, **per‑doc cap C=3**, **doc‑aware** Kdocs evaluation.

---

## 0) Design Principles & Controls
- **Fixed:** E1‑best chunking; tokenizer; PRAGMAs; ANN params for dense; BM25 defaults; doc‑aware ranking; evaluation.  
- **Variable:** retrieval mode (sparse/dense/hybrid) and fusion strategy.  
- **Primary metric:** NDCG@10 (graded); also MAP, MRR@10, Recall@k.

---

## 1) Corpora & Layout
E1 triad: **TREC‑COVID**, **SciFact**, **FiQA‑2018** (BEIR). Same validation as E1.

---

## 2) Retrieval Modes
- **BM25** only.  
- **Dense** only (E2 best embedder).  
- **Hybrid (α‑mix)**: standardize scores (z‑score per run), compute `α * dense + (1-α) * bm25` per chunk.  
- **Hybrid (RRF)**: fuse ranks from dense and BM25 (RRF with k=60).  
- **Hybrid (DAT)**: decide α per query using a lightweight **gating function**:
  - **Heuristic DAT**: α=0.2 if query has ≥2 proper nouns or exact numbers; else α=0.8.  
  - **LLM DAT (optional)**: small local LLM scores “lexicalness” vs “semanticness” (costs +10–50 ms).

**Candidate handling & doc ranking** identical to E1: **Nchunks=100**, **C=3**, fold chunks→docs, rank unique docs, evaluate **Kdocs=10**.

---

## 3) Metrics & Ledger
- **Quality:** NDCG@10, MAP, MRR@10, Recall@{1,5,10,20}.  
- **Cost:** retrieval latency p50/p95 per mode, **RSS peak**, **dup@N**, #chunks contributing, fusion overhead (ms).

Emit per variant: `results/e3_<dataset>/<mode>/{metrics.json,per_query.jsonl,ledger.jsonl}`.

---

## 4) Smoke Test
- Dataset: **TREC‑COVID**. Modes: **BM25**, **Dense**, **RRF**, **α‑mix(0.5)**. Optional: **DAT‑heuristic**.  
- Run on 50‑query stratified set; confirm metrics, overheads recorded.

---

## 5) Full Evaluation (Per Dataset)
- Modes: **BM25**, **Dense**, **RRF**, **α‑mix(0.2,0.5,0.8)**, **DAT‑heuristic**.  
- Procedure: run full queries, doc‑aware ranking, compute metrics & ledger.

**Statistics:** paired tests vs **best single** (BM25 or Dense) and vs **RRF**. Report mean ±95% CI, p‑values.  
**Ablations:** per‑query‑type breakdowns (entity‑heavy vs paraphrastic; use simple heuristics).

---

## 6) Operational Notes & Failure Modes
- **Score scaling:** For α‑mix, use **z‑score** per run to prevent scale mismatch; log μ/σ used.  
- **DAT stability:** Start with heuristic DAT; if LLM‑DAT added, cache gate outputs.  
- **Index parity:** Ensure BM25 index covers exactly the same doc set and tokenization cleanup as dense.

---

## 7) Work Items
- Implement fusion functions: z‑mix, RRF; optional DAT gate (heuristic + LLM).  
- CLI: `--retrieval {bm25,dense,hybrid_rrf,hybrid_amix,hybrid_dat}` and `--alpha`.  
- Eval runner `scripts/e3_eval.py` to orchestrate runs and plots.

---

## 8) Checklists
- [ ] E1/E2 baselines frozen; BM25 index built  
- [ ] Smoke test on TREC‑COVID done  
- [ ] Full grid per dataset; stats produced; per‑type analysis table

---

## 9) Command Snippets
```bash
# RRF on TREC-COVID
python scripts/e3_eval.py --dataset trec_covid \
  --retrieval hybrid_rrf --kdocs 10 --nchunks 100 --per_doc_cap 3

# α-mix (0.5)
python scripts/e3_eval.py --dataset trec_covid \
  --retrieval hybrid_amix --alpha 0.5 --kdocs 10
```
