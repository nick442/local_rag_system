# E6 — Context Budget & Top‑K Selection Execution Plan (Revised)

**Goal**  
Identify **optimal Kdocs** (and chunk ordering) to feed into the LLM, balancing **answer quality** and **latency**. Probe **position sensitivity** (“lost‑in‑the‑middle”) and diminishing returns of larger K on a local 4B LLM.

**Key Decisions**  
- Input contexts built from **doc‑aware ranked** retrieval outputs (from best E3/E4 pipeline).  
- Evaluate **retrieval‑conditioned answerability** and, where possible, **end‑to‑end EM/F1**.  
- Test **ordering**: relevance‑descending vs random permutation; optional middle‑placement stress test.

---

## 0) Design Principles & Controls
- **Fixed:** E1 chunking; best retrieval+rerank from E3/E4; same prompt template; same max context tokens.  
- **Variable:** **Kdocs ∈ {2,4,6,8,12}** and **ordering policy**.  
- Keep LLM model constant (e.g., Gemma‑3‑4B Q4_0) and decoding params fixed.

---

## 1) Corpora & Splits
- Primary: **SciFact** and **FiQA‑2018** (short answers / evidence helpful).  
- Optional: **NQ‑lite** (subset with gold answers) for strict EM/F1.  
- Use official BEIR queries; for NQ‑lite, provide gold answers; split dev/test to avoid tuning on test.

---

## 2) Context Construction
- From the best pipeline, obtain **top‑Kdocs** (doc‑aware; K variable).  
- Build the LLM context as concatenation of chunks with a **relevance header** (doc title + brief snippet).  
- **Ordering** variants:  
  - **Desc‑relevance** (default)  
  - **Random** (control)  
  - *(Optional)* **Middle‑placement**: force the most relevant chunk to the **middle** to test position effects.

Respect a **max token budget** (e.g., 2k) and truncate on the **least relevant** tails.

---

## 3) Metrics
- **Retrieval‑conditioned answerability (RCA):** fraction of queries where **any** provided chunk contains gold answer span (proxy for sufficiency).  
- **End‑to‑end** (where gold answers exist): **EM/F1**; also **faithfulness** proxy (citation rate: does the answer cite a retrieved chunk ID).  
- **Cost:** total generation time/query, tokens‑in/tokens‑out, and overall latency.

Emit per K & order: `results/e6_<dataset>/<K>_{desc,random,mid}/{metrics.json,per_query.jsonl,ledger.jsonl}`.

---

## 4) Smoke Test
- Dataset: **SciFact**; K={2,6}; ordering={desc,random}.  
- Check RCA rises with K; EM/F1 (if available) does not collapse; latency grows modestly.

---

## 5) Full Evaluation
- Datasets: SciFact, FiQA‑2018; *(optional)* NQ‑lite for EM/F1.  
- Kdocs: {2,4,6,8,12}; ordering: {desc,random,(mid)}.  
- Stats: paired tests across K vs **K=4** reference for RCA and EM/F1 (on overlapping query sets).  
- Plot **quality vs latency** curves to identify Pareto‑optimal K.

---

## 6) Operational Notes & Failure Modes
- **Context overflow:** dynamic budgeting—drop least‑relevant chunks first; log dropped counts.  
- **Citation extraction:** tag each chunk with an ID; instruct LLM to cite IDs in answers to compute faithfulness proxy.  
- **LLM variance:** run each setting with **2 seeds** (temperature kept constant or low, e.g., 0.2) and average results.

---

## 7) Work Items
- Context builder with ordering policies & budgeting.  
- RCA scorer (string match / span overlap within chunks).  
- Simple EM/F1 script for NQ‑lite; citation parser.  
- Eval runner `scripts/e6_eval.py` to sweep K and generate plots.

---

## 8) Checklists
- [ ] Best pipeline from E3/E4 frozen  
- [ ] Context builder outputs include chunk IDs and titles  
- [ ] Smoke test passes (RCA increases with K)  
- [ ] Full sweep done; Pareto K selected and justified

---

## 9) Command Snippets
```bash
# Sweep K on SciFact (desc ordering)
python scripts/e6_eval.py --dataset scifact --order desc \
  --kdocs_list 2 4 6 8 12 --max_ctx_tokens 2000

# Random-order control
python scripts/e6_eval.py --dataset scifact --order random \
  --kdocs_list 2 6
```
