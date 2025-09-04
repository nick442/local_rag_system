# E4 — Lightweight Reranking Execution Plan (Revised)

**Goal**  
Measure quality gains and latency costs from **CPU‑friendly cross‑encoder rerankers** on top of the best first‑stage retriever (from E3). Compare **MiniLM‑L6‑v2** vs **BGE‑reranker‑v2‑m3** and select a pragmatic default for 16 GB.

**Key Decisions**  
- First stage: **best hybrid/dense** from E3 (fixed). Candidate pool **Nchunks=100**.  
- Rerankers: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-v2-m3`.  
- Batch sizes tuned for CPU/MPS (8–16).  
- **Doc‑aware** evaluation at **Kdocs=10**; measure ΔNDCG@10 and Recall@1.

---

## 0) Design Principles & Controls
- **Fixed:** E1 chunking; E2 embedder; E3 mode; tokenizer; PRAGMAs; ANN params.  
- **Variable:** reranker choice and candidate‑k for reranking (Kcand ∈ {20,50,100}).

---

## 1) Corpora
E1 triad: **TREC‑COVID**, **SciFact**, **FiQA‑2018** (BEIR).

---

## 2) Pipeline
1) Retrieve **top‑Nchunks** with first stage (N=100).  
2) **Per‑doc cap C=3** (or MMR for diversity).  
3) Build **pairs**: (query, chunk_text) for chunks in pool.  
4) **Rerank** with cross‑encoder (batch size 8–16).  
5) Fold to **unique docs** (best chunk per doc), **rank docs**, evaluate **Kdocs=10**.

---

## 3) Variants
- **No‑rerank** (baseline)  
- **MiniLM‑L6‑v2** rerank  
- **BGE‑reranker‑v2‑m3** rerank  
- *(Optional)* **Top‑Kcand sensitivity**: rerank @20 and @50 vs @100.

---

## 4) Metrics & Ledger
- **ΔNDCG@10**, **Recall@1** improvement vs baseline; **MRR@10**, Recall@{5,10,20}.  
- **Latency**: rerank time/query (ms), total time/query; **throughput** (pairs/s).  
- **RSS peak (MB)**; **#pairs** evaluated; **dup@N** of candidate pool.

Emit per variant: `results/e4_<dataset>/<variant>/{metrics.json,per_query.jsonl,ledger.jsonl}`.

---

## 5) Smoke Test
- Dataset: **TREC‑COVID**; candidate **N=50**; run **MiniLM** rerank; confirm ΔNDCG>0 and latency reasonable.  
- Spot‑check top‑ranked passages for relevance.

---

## 6) Full Evaluation (Per Dataset)
- Variants: baseline, MiniLM, BGE‑v2‑m3. Candidate **N=100**; batch size 8–16.  
- Stats: paired tests vs baseline and between rerankers; 95% CI.  
- Report best **quality vs latency** trade‑off and recommended default (e.g., MiniLM@N=50).

---

## 7) Operational Notes & Failure Modes
- **CPU only** is acceptable; if MPS acceleration is possible for inference, record it.  
- If latency too high, reduce candidate **N** or batch size; report trade‑offs explicitly.  
- Cache tokenization to avoid overhead during rerank.

---

## 8) Work Items
- Reranker wrapper with batch inference & pair building; CLI: `--rerank {none,minilm,bge_v2_m3} --kcand 50`.  
- Eval runner `scripts/e4_eval.py` to orchestrate runs, compute Δs, and generate plots.

---

## 9) Checklists
- [ ] First‑stage (from E3) locked; candidate N validated  
- [ ] Both rerankers downloaded & smoke tested  
- [ ] Full runs per dataset; stats/plots done; recommendation recorded

---

## 10) Command Snippets
```bash
# Rerank @50 with MiniLM
python scripts/e4_eval.py --dataset trec_covid \
  --retrieval best_from_e3 --rerank minilm --kcand 50 --kdocs 10

# Compare with BGE reranker
python scripts/e4_eval.py --dataset trec_covid \
  --retrieval best_from_e3 --rerank bge_v2_m3 --kcand 50 --kdocs 10
```
