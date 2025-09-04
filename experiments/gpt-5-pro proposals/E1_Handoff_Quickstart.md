# E1 Handoff Quickstart (for a new agent)

Purpose: get you executing the revised E1 (Chunking) plan quickly and safely on this repo.

---

## 1) Branch & PR Workflow

- Base branch: `GPT-5-Experiments` (do not touch `main`).
- Create feature branches from `GPT-5-Experiments` and open PRs with base set to `GPT-5-Experiments`.
- Example:
  - `git fetch origin`
  - `git checkout GPT-5-Experiments && git pull`
  - `git checkout -b feature/e1-your-task`

Reference: policy enforced in `AGENTS.md`.

---

## 2) Environment

Always run Python through the project conda env:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python -V
```

Useful status checks:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python main.py status
```

---

## 3) Dataset Prep (TREC‑COVID)

We evaluate on BEIR’s TREC‑COVID split. Local layout (not committed):

```
datasets/trec_covid/
  docs/               # one file per doc; name each file as <docid>.txt
  queries.jsonl       # {"qid": str, "text": str}
  qrels.tsv           # qid \t 0 \t docid \t relevance
```

Important: name each doc file by its BEIR `docid` (e.g., `12345.txt`). The ingestion pipeline records `filename` in metadata; we use that to map retrieval results back to BEIR doc IDs for evaluation.

Acceptance before proceeding:
- qrels doc IDs are all present in `docs/` (i.e., `qrels.docids ⊆ docs_filenames`).

---

## 4) Ingest Per‑Variant Collections

We isolate chunking effects by ingesting each variant into its own `collection_id`. The CLI already supports chunk flags.

Examples (fixed variants):

```bash
# 256/20
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_fixed_256_20 --chunk-size 256 --chunk-overlap 20

# 512/50
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py ingest directory datasets/trec_covid/docs \
  --collection trec_covid_fixed_512_50 --chunk-size 512 --chunk-overlap 50
```

Sanity:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python main.py collection list && python main.py analytics stats --collection trec_covid_fixed_256_20
```

---

## 5) Create a Stratified 50‑Query Smoke Set

If you don’t have `queries_small.jsonl`, create a deterministic subset now (simple first‑N fallback shown):

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python - <<'PY'
import json, pathlib
qp = pathlib.Path('datasets/trec_covid/queries.jsonl')
out = pathlib.Path('datasets/trec_covid/queries_small.jsonl')
lines = qp.read_text().splitlines()
out.write_text('\n'.join(lines[:50]))
print(f'wrote {out} with 50 queries')
PY
```

---

## 6) Quick Evaluation Runner (doc‑aware ranking)

Use this one‑off to run the smoke test end‑to‑end (dense retrieval only) with **doc‑aware ranking** (cap 3 chunks/doc → fold to docs → top‑10 docs). It writes metrics + per‑query outputs.

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
python - <<'PY'
import os, json, pathlib, sqlite3
from pathlib import Path
from collections import defaultdict

# Ensure project src/ is on sys.path
import sys
sys.path.insert(0, str(Path('.').resolve()))

from src.rag_pipeline import RAGPipeline
from src.evaluation_metrics import RetrievalQualityEvaluator

DB_PATH = 'data/rag_vectors.db'
EMB_PATH = 'models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
LLM_PATH = os.getenv('LLM_MODEL_PATH', 'models/gemma-3-4b-it-q4_0.gguf')
DATASET = 'trec_covid'
QUERIES = f'datasets/{DATASET}/queries_small.jsonl'
QRELS = f'datasets/{DATASET}/qrels.tsv'
COLLECTION = 'trec_covid_fixed_256_20'  # change to your target collection
OUTDIR = Path(f'results/e1_{DATASET}/test/{COLLECTION}')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Load queries
queries = []
with open(QUERIES, 'r') as f:
    for line in f:
        obj = json.loads(line)
        queries.append((obj['qid'], obj['text']))

# Load qrels
truth = defaultdict(dict)
with open(QRELS, 'r') as f:
    for line in f:
        qid, _zero, docid, rel = line.strip().split('\t')
        truth[qid][docid] = float(rel)

# Init RAG (we'll use retriever only; no generation)
rag = RAGPipeline(db_path=DB_PATH, embedding_model_path=EMB_PATH, llm_model_path=LLM_PATH)

KDOCS = 10
NCHUNKS = 100
PER_DOC_CAP = 3

results_docs = {}

for qid, text in queries:
    # retrieve chunks
    chunks = rag.retriever.retrieve(text, k=NCHUNKS, method='vector', collection_id=COLLECTION)
    # doc-aware pool (limit chunks/doc, keep best chunk score per doc)
    pool = defaultdict(list)
    for ch in chunks:
        # filename holds original BEIR docid; we strip extension
        meta = ch.metadata or {}
        fname = meta.get('filename') or meta.get('source') or 'Unknown'
        doc_key = Path(fname).stem
        if doc_key == 'Unknown':
            continue
        if len(pool[doc_key]) < PER_DOC_CAP:
            pool[doc_key].append(ch)
    # best-chunk score per doc
    doc_scores = []
    for doc_key, items in pool.items():
        best = max(items, key=lambda r: r.score)
        doc_scores.append((doc_key, best.score))
    # rank docs and cut to KDOCS
    ranked_docs = [d for d,_ in sorted(doc_scores, key=lambda x: x[1], reverse=True)][:KDOCS]
    results_docs[qid] = ranked_docs

# Evaluate
rqe = RetrievalQualityEvaluator(ground_truth_relevance=truth)
metrics = rqe.evaluate_query_set(results_docs, k_values=[1,5,10,20])

# Persist
(OUTDIR / 'metrics.json').write_text(json.dumps(metrics, indent=2))
print(f'Wrote {OUTDIR / 'metrics.json'}')
PY
```

Change `COLLECTION` to evaluate other ingested variants (e.g., `trec_covid_fixed_512_50`). Repeat to compare outputs.

---

## 7) Full Grid (when ready)

Follow the revised plan in `E1_Chunking_Execution_Plan.md`:
- Ingest 9 fixed variants (sizes {128,256,512} × overlaps {0,20,50}).
- Use the evaluation script (or extend the quick runner) on the full query set.
- Produce metrics + ledger per variant; run paired tests vs `fixed_256_20`.

---

## 8) Troubleshooting

- sqlite‑vec fails to load: the system falls back to manual similarity; expect slower queries. Continue and note it in results.
- Embedding dim mismatch: ensure we’re using 384‑dim MiniLM for E1 (or reindex in a new DB).
- Doc ID mismatch: be sure doc filenames in `docs/` are the canonical BEIR doc IDs; evaluation maps by filename stem.

---

## 9) Quick Checklist

- [ ] Branch from `GPT-5-Experiments`; PRs base on `GPT-5-Experiments`.
- [ ] Dataset at `datasets/trec_covid/` with filenames = BEIR doc IDs.
- [ ] Ingest at least two collections (e.g., `trec_covid_fixed_256_20`, `trec_covid_fixed_512_50`).
- [ ] Create `queries_small.jsonl` (50 queries).
- [ ] Run the quick evaluation runner for both collections; inspect `metrics.json`.
- [ ] Record any deviations (PRAGMAs, sqlite‑vec status) with the results.
