# E5 — LLM‑Aided Query Expansion Execution Plan (Revised)

**Goal**  
Evaluate **HyDE**, **Query2Doc**, and **Corpus‑Steered Query Expansion (CSQE)** for improving retrieval on BEIR tasks under local constraints. Quantify gains across **BM25**, **Dense**, and **Hybrid** first stages, and measure expansion **costs** and **leakage risk**.

**Key Decisions**  
- Expansions generated with the **local LLM** (Gemma‑3‑4B‑IT Q4) to keep everything offline.  
- **Caps**: expansion ≤ **256 tokens**; caching enabled.  
- Compare **no‑expansion** vs **HyDE**, **Query2Doc**, **CSQE**.  
- Evaluate on top of **BM25**, **Dense (E2 best)**, and **Hybrid (E3 best RRF)**.  
- Doc‑aware ranking; Kdocs=10; Nchunks=100; C=3.

---

## 0) Design Principles & Controls
- **Fixed:** E1 chunking; E2/E3 best retrievers; tokenizer; PRAGMAs.  
- **Variable:** expansion strategy and first‑stage retrieval mode.

---

## 1) Corpora
E1 triad (TREC‑COVID, SciFact, FiQA‑2018).

---

## 2) Expansion Methods
- **HyDE:** prompt LLM to produce a “hypothetical answer/document”; embed & retrieve with it (optionally combine with original query embedding).  
- **Query2Doc:** few‑shot prompt to synthesize a pseudo‑document that would answer the query; use as retrieval text (boosts BM25).  
- **CSQE:** extract top‑sentences from **initial in‑corpus retrieval** (e.g., BM25 top‑10); concatenate and lightly rewrite with LLM; use as expansion (minimizes leakage).

**Caching:** store expansions by `(dataset, qid, method)`; reuse across modes.

---

## 3) Retrieval Modes with Expansion
For each expansion method (including none), run:
- **BM25**: query text = expansion (or original)  
- **Dense**: embed expansion (or original)  
- **Hybrid (RRF)**: fuse BM25 & Dense runs derived from the same expansion

Candidate handling and **doc‑aware ranking** identical to E1 (Nchunks=100, C=3, Kdocs=10).

---

## 4) Metrics & Costs
- **Quality:** NDCG@10, MAP, MRR@10, Recall@{1,5,10,20}.  
- **Cost:** expansion time/query (ms), token count, % cache hits, added retrieval latency vs baseline.  
- **Risk proxy:** for HyDE/Query2Doc, log **out‑of‑corpus n‑gram rate** (fraction of expansion tokens unseen in corpus vocabulary) as a crude leakage indicator.

Emit per variant: `results/e5_<dataset>/<mode>_{none,hyde,q2d,csqe}/*`.

---

## 5) Smoke Test
- Dataset: **FiQA‑2018** (noisy queries). Run **BM25** with **none** vs **q2d**; confirm NDCG lift and cost profile.  
- Ensure expansion cache working; cap length enforced.

---

## 6) Full Evaluation (Per Dataset)
- Modes: BM25, Dense, Hybrid(RRF).  
- Expansions: none, HyDE, Query2Doc, CSQE.  
- Stats: paired tests vs **none** and between methods; 95% CI.  
- Report **best expansion per mode** and whether gains persist after Hybrid+Rerank (from E4).

---

## 7) Operational Notes & Failure Modes
- **Token budget:** keep ≤256 to bound latency; truncate nicely at sentence boundaries.  
- **Bad expansions:** if LLM produces unrelated text, fall back to original query (log fallback).  
- **Local limits:** if LLM infra is busy, run expansions offline ahead of eval (cache‑first).

---

## 8) Work Items
- Implement expansion generators (HyDE, Q2D, CSQE) with prompts & caching.  
- CLI: `--expand {none,hyde,q2d,csqe} --expand-max-tokens 256`.  
- Eval runner `scripts/e5_eval.py` looping (mode × expansion).

---

## 9) Checklists
- [ ] LLM expansion endpoints wired & cached  
- [ ] Smoke test on FiQA complete  
- [ ] Full grid per dataset; stats/plots done; leakage proxy computed

---

## 10) Command Snippets
```bash
# Hybrid with CSQE on SciFact
python scripts/e5_eval.py --dataset scifact \
  --retrieval hybrid_rrf --expand csqe --kdocs 10 --nchunks 100 --per_doc_cap 3

# Compare BM25 none vs q2d on FiQA
python scripts/e5_eval.py --dataset fiqa \
  --retrieval bm25 --expand none --kdocs 10
python scripts/e5_eval.py --dataset fiqa \
  --retrieval bm25 --expand q2d --kdocs 10
```
