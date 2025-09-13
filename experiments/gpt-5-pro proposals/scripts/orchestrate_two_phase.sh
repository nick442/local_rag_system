#!/usr/bin/env bash
set -euo pipefail

# Orchestrate two-phase ingest: waits for ingest PID to finish, then re-embeds, rebuilds, and vacuums.
#
# Options:
#   -d  DB path (e.g., data/rag_vectors__trec_covid_fixed_256_20.db)
#   -c  Collection ID (e.g., trec_covid_fixed_256_20)
#   -e  Embedding model path (SentenceTransformers local path)
#   -p  Ingest PID file path (written by the background ingest)
#   -b  Re-embed batch size (default: 32)
#   -r  Repo root (for running main.py), default: current directory
#   -l  Log file for this orchestrator (optional)

DB=""
COLL=""
EMB_PATH=""
PID_FILE=""
BATCH_SIZE=32
REPO_ROOT="$(pwd -P)"
LOG_FILE=""

usage() {
  echo "Usage: $0 -d <db_path> -c <collection> -e <embedding_path> -p <pid_file> [-b <batch>] [-r <repo_root>] [-l <log>]" >&2
}

while getopts "d:c:e:p:b:r:l:" opt; do
  case "$opt" in
    d) DB="$OPTARG";;
    c) COLL="$OPTARG";;
    e) EMB_PATH="$OPTARG";;
    p) PID_FILE="$OPTARG";;
    b) BATCH_SIZE="$OPTARG";;
    r) REPO_ROOT="$OPTARG";;
    l) LOG_FILE="$OPTARG";;
    *) usage; exit 2;;
  esac
done

if [[ -z "$DB" || -z "$COLL" || -z "$EMB_PATH" || -z "$PID_FILE" ]]; then
  usage; exit 2
fi

cd "$REPO_ROOT"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { if [[ -n "${LOG_FILE}" ]]; then echo "[$(ts)] $*" | tee -a "$LOG_FILE"; else echo "[$(ts)] $*"; fi }

log "Orchestrator started for collection '$COLL' using DB '$DB'"

# Wait for PID file to appear
TRIES=0
until [[ -f "$PID_FILE" ]]; do
  ((TRIES++))
  if (( TRIES > 120 )); then
    log "Timed out waiting for PID file: $PID_FILE"
    exit 1
  fi
  sleep 1
done

PID="$(cat "$PID_FILE" || true)"
if [[ -z "${PID}" ]]; then
  log "PID file is empty: $PID_FILE"; exit 1
fi

log "Monitoring ingest process PID $PID ..."
while ps -p "$PID" >/dev/null 2>&1; do
  sleep 10
done
log "Ingest process PID $PID finished. Starting re-embed..."

# Activate env and run re-embed, rebuild, vacuum
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rag_env

set +e
python main.py --db-path "$DB" maintenance reindex --operation reembed --collection "$COLL" \
  --embedding-path "$EMB_PATH" --batch-size "$BATCH_SIZE" --no-backup >> "${LOG_FILE:-/dev/null}" 2>&1
REEMBED_RC=$?
set -e
if [[ $REEMBED_RC -ne 0 ]]; then
  log "Re-embed failed with code $REEMBED_RC"; exit $REEMBED_RC
fi
log "Re-embed completed. Rebuilding indices..."

python main.py --db-path "$DB" maintenance reindex --operation rebuild >> "${LOG_FILE:-/dev/null}" 2>&1 || true
log "Rebuild done. Vacuuming..."
python main.py --db-path "$DB" maintenance reindex --operation vacuum >> "${LOG_FILE:-/devnull}" 2>&1 || true

log "Final stats:"
python main.py --db-path "$DB" analytics stats --collection "$COLL" >> "${LOG_FILE:-/dev/null}" 2>&1 || true
log "Orchestration complete."

exit 0

