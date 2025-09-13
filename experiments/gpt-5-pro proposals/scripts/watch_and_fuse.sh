#!/usr/bin/env bash
set -euo pipefail

# Watch for per_query outputs from e1_eval, then run E3 fusion once available.
# Usage:
#   source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
#   bash experiments/gpt-5-pro proposals/scripts/watch_and_fuse.sh \
#     <dataset_dir> <outdir> <db_path> <collection> <embedding_path>

DS_DIR="$1"      # e.g., experiments/gpt-5-pro proposals/datasets/scifact
OUTDIR="$2"      # e.g., results/e1_scifact/full/scifact_fixed_512_50
DB_PATH="$3"     # e.g., data/rag_vectors__scifact_fixed_512_50.db
COLL="$4"        # e.g., scifact_fixed_512_50
EMB="$5"         # embedding path

LOG_ROOT="results/e1_triad/logs"
mkdir -p "$LOG_ROOT"
FLOG="$LOG_ROOT/fusion_${COLL}.log"

echo "[watch_and_fuse] Watching $OUTDIR for per_query files (COLL=$COLL)" | tee -a "$FLOG"

attempts=0
while true; do
  if [ -f "$OUTDIR/per_query.jsonl" ] && [ -f "$OUTDIR/per_query_bm25.jsonl" ]; then
    echo "[watch_and_fuse] Found per_query files. Running fusion for $COLL" | tee -a "$FLOG"
    python "experiments/gpt-5-pro proposals/scripts/e3_fusion.py" \
      --dataset-dir "$DS_DIR" --outdir "$OUTDIR" \
      --db-path "$DB_PATH" --embedding-path "$EMB" --collection "$COLL" \
      --kdocs 10 --nchunks 100 --cap 3 --rrf_k 60 >> "$FLOG" 2>&1 || true
    echo "[watch_and_fuse] Fusion complete for $COLL" | tee -a "$FLOG"
    break
  fi
  attempts=$((attempts+1))
  if [ $attempts -gt 720 ]; then
    echo "[watch_and_fuse] Timeout waiting for per_query files (12h). Exiting." | tee -a "$FLOG"
    exit 1
  fi
  sleep 60
done

