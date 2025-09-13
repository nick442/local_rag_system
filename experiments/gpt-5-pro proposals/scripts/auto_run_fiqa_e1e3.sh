#!/usr/bin/env bash
set -euo pipefail

# Auto-run FiQA 512/50 E1 (dense/bm25/hybrid) + E3 fusion after ingestion completes
# - Waits for existing ingestion PID to exit (if present)
# - Rebuilds indices
# - Runs e1_eval for dense, bm25, hybrid with diagnostics
# - Runs e3_fusion (RRF + Z-score)
# - Writes aliased E1 metrics matching corpus doc IDs
# - Produces combined SciFact+FiQA summary JSON

CONDA_PREFIX_CMD="source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env"

EMB="models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

FIQA_DB="data/rag_vectors__fiqa_fixed_512_50.db"
FIQA_DS_DIR="experiments/gpt-5-pro proposals/datasets/fiqa"
FIQA_OUTDIR="results/e1_fiqa/full/fiqa_fixed_512_50"

SCIFACT_DB="data/rag_vectors__scifact_fixed_512_50.db"
SCIFACT_OUTDIR="results/e1_scifact/full/scifact_fixed_512_50"

LOG_DIR="results/e1_triad/logs"
mkdir -p "$LOG_DIR" "$FIQA_OUTDIR"

LOG_FILE="$LOG_DIR/fiqa_fixed_512_50.auto_e1e3.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting auto_run_fiqa_e1e3.sh" | tee -a "$LOG_FILE"

# 1) Wait for ingestion to finish if a PID file exists
PIDF="$LOG_DIR/fiqa_fixed_512_50.ingest.pid"
if [[ -f "$PIDF" ]]; then
  PID=$(cat "$PIDF" || true)
  if [[ -n "${PID}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for ingestion PID $PID to exit..." | tee -a "$LOG_FILE"
    while ps -p "$PID" > /dev/null 2>&1; do
      sleep 60
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ingestion process $PID finished" | tee -a "$LOG_FILE"
  fi
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] No ingestion PID file found; continuing" | tee -a "$LOG_FILE"
fi

# 2) Rebuild indices
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rebuilding indices for FiQA DB" | tee -a "$LOG_FILE"
eval "$CONDA_PREFIX_CMD && python main.py --db-path '$FIQA_DB' maintenance reindex --operation rebuild" | tee -a "$LOG_FILE"

# 3) Run E1 evaluations (dense, bm25, hybrid) with diagnostics
E1_EVAL="experiments/gpt-5-pro proposals/scripts/e1_eval.py"
for MODE in dense bm25 hybrid; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] E1 eval ($MODE) for FiQA" | tee -a "$LOG_FILE"
  eval "$CONDA_PREFIX_CMD && python '$E1_EVAL' \
    --db-path '$FIQA_DB' \
    --embedding-path '$EMB' \
    --collection 'fiqa_fixed_512_50' \
    --dataset-dir '$FIQA_DS_DIR' \
    --mode '$MODE' \
    --kdocs 10 --nchunks 100 --cap 3 \
    --emit_diagnostics \
    --outdir '$FIQA_OUTDIR'" | tee -a "$LOG_FILE"
done

# 4) Write aliased metrics for FiQA (map internal doc_id -> corpus filename stem)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing aliased FiQA metrics" | tee -a "$LOG_FILE"
eval "$CONDA_PREFIX_CMD && python - << 'PY'" | tee -a "$LOG_FILE"
import json, sqlite3
from pathlib import Path

db_path = Path('data/rag_vectors__fiqa_fixed_512_50.db')
outdir = Path('results/e1_fiqa/full/fiqa_fixed_512_50')
coll = 'fiqa_fixed_512_50'

aliases = {}
with sqlite3.connect(str(db_path)) as con:
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT doc_id, source_path FROM documents WHERE collection_id = ?", (coll,))
    for row in cur.fetchall():
        src = row['source_path'] or ''
        stem = Path(src).stem
        aliases[row['doc_id']] = stem

def load_pq(path: Path):
    res = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            o = json.loads(line)
            res[o['qid']] = [aliases.get(d, d) for d in o.get('ranked_docs', [])]
    return res

pq_dense = load_pq(outdir / 'per_query.jsonl')
pq_bm25 = load_pq(outdir / 'per_query_bm25.jsonl')
pq_hybrid = load_pq(outdir / 'per_query_hybrid.jsonl')

from src.evaluation_metrics import RetrievalQualityEvaluator
qrels = {}
with open('experiments/gpt-5-pro proposals/datasets/fiqa/qrels.tsv', 'r', encoding='utf-8') as f:
    for raw in f:
        s=raw.strip()
        if not s: continue
        parts = s.split('\t')
        try:
            if len(parts)>=4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            elif len(parts)==3:
                qid, docid, rel = parts[0], parts[1], parts[2]
            else:
                continue
            rel_v = float(rel)
        except Exception:
            continue
        qrels.setdefault(qid, {})[docid] = rel_v

rqe = RetrievalQualityEvaluator(ground_truth_relevance=qrels)
res_dense = rqe.evaluate_query_set(pq_dense, k_values=[1,5,10,20])
res_bm25 = rqe.evaluate_query_set(pq_bm25, k_values=[1,5,10,20])
res_hybrid = rqe.evaluate_query_set(pq_hybrid, k_values=[1,5,10,20])

(outdir / 'metrics_dense_aliased.json').write_text(json.dumps(res_dense, indent=2))
(outdir / 'metrics_bm25_aliased.json').write_text(json.dumps(res_bm25, indent=2))
(outdir / 'metrics_hybrid_aliased.json').write_text(json.dumps(res_hybrid, indent=2))
print('Aliased FiQA E1 metrics written')
PY

# 5) Run E3 fusion for FiQA
E3_FUSION="experiments/gpt-5-pro proposals/scripts/e3_fusion.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running E3 fusion for FiQA" | tee -a "$LOG_FILE"
eval "$CONDA_PREFIX_CMD && python '$E3_FUSION' \
  --dataset-dir '$FIQA_DS_DIR' \
  --outdir '$FIQA_OUTDIR' \
  --db-path '$FIQA_DB' \
  --embedding-path '$EMB' \
  --collection 'fiqa_fixed_512_50' \
  --kdocs 10 --nchunks 100 --cap 3 --rrf_k 60" | tee -a "$LOG_FILE"

# 6) Combined SciFact + FiQA summary (E1/E3)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing combined summary" | tee -a "$LOG_FILE"
eval "$CONDA_PREFIX_CMD && python - << 'PY'" | tee -a "$LOG_FILE"
import json
from pathlib import Path

def pick(m):
    return {
        'P@1': float(m.get('precision_at_k',{}).get('1',{}).get('mean',0.0)),
        'P@10': float(m.get('precision_at_k',{}).get('10',{}).get('mean',0.0)),
        'NDCG@10': float(m.get('ndcg_at_k',{}).get('10',{}).get('mean',0.0)),
        'MRR': float(m.get('mrr',0.0)),
    }

def load_latency(p):
    try:
        o=json.load(open(p));
        return {k: round(o.get(k,0.0),2) for k in ['mean_ms','p50_ms','p95_ms']}
    except Exception:
        return {}

def load_position(p):
    try:
        o=json.load(open(p));
        return {
          'mean_top1_relative_position': round(o.get('mean_top1_relative_position',0.0),3),
          'n': o.get('n',0),
          'histogram_top1': o.get('histogram_top1',{}),
        }
    except Exception:
        return {}

def dataset_block(root):
    r=Path(root)
    out={}
    # E1 aliased metrics
    for mode,fn in [('dense','metrics_dense_aliased.json'),('bm25','metrics_bm25_aliased.json'),('hybrid','metrics_hybrid_aliased.json')]:
        p=r/fn
        if p.exists():
            out[mode]=pick(json.load(open(p)))
    # E3 fusion
    for name,fn in [('fusion_rrf','metrics_rrf.json'),('fusion_zs','metrics_zscore.json')]:
        p=r/fn
        if p.exists():
            out[name]=pick(json.load(open(p)))
    # Latency
    for mode,fn in [('dense','latency.json'),('bm25','latency_bm25.json'),('hybrid','latency_hybrid.json')]:
        p=r/fn
        if p.exists():
            out.setdefault('latency',{})[mode]=load_latency(p)
    # Position
    for mode,fn in [('dense','position.json'),('bm25','position_bm25.json'),('hybrid','position_hybrid.json')]:
        p=r/fn
        if p.exists():
            out.setdefault('position',{})[mode]=load_position(p)
    return out

scifact = dataset_block('results/e1_scifact/full/scifact_fixed_512_50')
fiqa = dataset_block('results/e1_fiqa/full/fiqa_fixed_512_50')

def lift(a,b,key='NDCG@10'):
    try:
        return round((b.get(key,0.0) - a.get(key,0.0)),6)
    except Exception:
        return 0.0

summary={
  'scifact_fixed_512_50': scifact,
  'fiqa_fixed_512_50': fiqa,
  'notes': {
    'fusion_lift_scifact_rrf_minus_dense': lift(scifact.get('dense',{}), scifact.get('fusion_rrf',{})),
    'fusion_lift_fiqa_rrf_minus_dense': lift(fiqa.get('dense',{}), fiqa.get('fusion_rrf',{})),
  }
}

outdir=Path('results/e1_triad/summary')
outdir.mkdir(parents=True, exist_ok=True)
(outdir/'summary_scifact_fiqa_512_50.json').write_text(json.dumps(summary, indent=2))
print('Combined summary written to', outdir/'summary_scifact_fiqa_512_50.json')
PY

echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_run_fiqa_e1e3.sh completed" | tee -a "$LOG_FILE"

