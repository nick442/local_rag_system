#!/usr/bin/env bash
set -euo pipefail

# E1 Triad Orchestrator with Live Tracking
# - Datasets: SciFact + FiQA
# - Variants: fixed 256/20, fixed 512/50, semantic 256/0
# - Two-phase ops: no-embed -> reembed -> rebuild -> vacuum
# - Live tracking: queue view + per-step progress, stats polling, per-variant logs

# Usage:
#   bash "experiments/gpt-5-pro proposals/scripts/run_e1_triad.sh"

# Conda env prefix (repo guideline)
CONDA_PREFIX_CMD="source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env"

# Threading guard (stability on some BLAS/OpenMP stacks) - allow override
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export KMP_INIT_AT_FORK="FALSE"
export OMP_WAIT_POLICY="PASSIVE"
export KMP_AFFINITY="disabled"

# Quieter downstream tools
export TQDM_DISABLE="1"
export HF_HUB_DISABLE_PROGRESS="1"
export TRANSFORMERS_VERBOSITY="error"
export TOKENIZERS_PARALLELISM="false"
export PYTHONWARNINGS="ignore"
export RICH_NO_COLOR="1"
export RICH_FORCE_TERMINAL="false"

# Raise file descriptor limit to avoid 'Too many open files' during ingest
ulimit -n 16384 2>/dev/null || ulimit -n 8192 2>/dev/null || true

EMB="models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
SCIDIR="experiments/gpt-5-pro proposals/datasets/scifact"
FIQDIR="experiments/gpt-5-pro proposals/datasets/fiqa"
EVAL="experiments/gpt-5-pro proposals/scripts/e1_eval.py"

RUN_ROOT="results/e1_triad"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Tunables (can be overridden via env)
FAST_MODE="${FAST_MODE:-0}"
INGEST_WORKERS="${INGEST_WORKERS:-4}"
REEMBED_BATCH="${REEMBED_BATCH:-64}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-15}"

log() {
  local msg="$1"; shift || true
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "$LOG_DIR/runner.log"
}

# Split a queue line into named variables (handles spaces in paths)
parse_line() {
  local s="$1"
  s=${s%$'\r'}
  NAME="${s%%|*}"; s="${s#*|}"
  DS_DIR="${s%%|*}"; s="${s#*|}"
  VAR="${s%%|*}"; s="${s#*|}"
  CHUNK_SIZE="${s%%|*}"; s="${s#*|}"
  CHUNK_OVERLAP="${s%%|*}"; s="${s#*|}"
  BACKEND="$s"
}

# Timing helpers
epoch() { date +%s; }

start_stage() {
  local COLL="$1"; local STAGE="$2"
  echo "$STAGE" > "$LOG_DIR/${COLL}.stage"
  echo "$STAGE START $(epoch)" >> "$LOG_DIR/${COLL}.timing"
}

end_stage() {
  local COLL="$1"; local STAGE="$2"
  echo "$STAGE END $(epoch)" >> "$LOG_DIR/${COLL}.timing"
}

stage_elapsed_secs() {
  local COLL="$1"; local STAGE="$2"
  local tf="$LOG_DIR/${COLL}.timing"
  [ -f "$tf" ] || { echo 0; return; }
  local start
  start=$(grep -E "^${STAGE} START " "$tf" | tail -n1 | awk '{print $3}')
  if [ -z "$start" ]; then echo 0; else echo $(( $(epoch) - start )); fi
}

fmt_dur() {
  local s=$1
  if [ -z "$s" ] || [ "$s" -le 0 ] 2>/dev/null; then echo "-"; return; fi
  local h=$(( s / 3600 ))
  local m=$(( (s % 3600) / 60 ))
  local sec=$(( s % 60 ))
  if [ "$h" -gt 0 ]; then
    printf "%dh%02dm%02ds" "$h" "$m" "$sec"
  else
    printf "%dm%02ds" "$m" "$sec"
  fi
}

avg_stage_duration() {
  # Sets globals AVG_STG_SECS and AVG_STG_COUNT
  local STAGE="$1"
  local TARGET="$STAGE"
  if [ "$STAGE" = "INGEST-RETRY" ]; then TARGET="INGEST"; fi
  local sum=0
  local cnt=0
  local dur
  for tf in "$LOG_DIR"/*.timing; do
    [ -f "$tf" ] || continue
    dur=$(awk -v ST="${TARGET}" '($1==ST && $2=="START"){st=$3} ($1==ST && $2=="END" && st>0){d=$3-st; if(d>0){last=d} st=0} END{if(last>0)print last; else print 0}' "$tf")
    if [ "$dur" -gt 0 ] 2>/dev/null; then
      sum=$((sum + dur))
      cnt=$((cnt + 1))
    fi
  done
  if [ "$cnt" -gt 0 ]; then
    AVG_STG_SECS=$((sum / cnt))
    AVG_STG_COUNT=$cnt
  else
    AVG_STG_SECS=0
    AVG_STG_COUNT=0
  fi
}

# Map stage name to an index 0..7 (7 means all stages done)
stage_index() {
  local s="$1"
  case "$s" in
    INGEST|INGEST-RETRY) echo 0 ;;
    BM25) echo 1 ;;
    REEMBED) echo 2 ;;
    REBUILD) echo 3 ;;
    VACUUM) echo 4 ;;
    DENSE) echo 5 ;;
    HYBRID) echo 6 ;;
    DONE) echo 7 ;;
    *) echo 0 ;;
  esac
}

# Compute overall progress across all variants (6 stages each)
compute_progress() {
  local total_stages=0
  local done_stages=0
  local Q_NAME Q_DIR Q_VAR Q_SIZE Q_OVERLAP Q_BACKEND
  while IFS='|' read -r Q_NAME Q_DIR Q_VAR Q_SIZE Q_OVERLAP Q_BACKEND; do
    [ -z "$Q_NAME" ] && continue
    total_stages=$((total_stages + 7))
    local COLL="${Q_NAME}_${Q_VAR}"
    if [ -f "$LOG_DIR/${COLL}.done" ]; then
      done_stages=$((done_stages + 6))
    else
      local st
      st=$(cat "$LOG_DIR/${COLL}.stage" 2>/dev/null || echo "")
      local idx
      idx=$(stage_index "$st")
      done_stages=$((done_stages + idx))
    fi
  done < "$LOG_DIR/queue.txt"
  PROG_DONE_STAGES=$done_stages
  PROG_TOTAL_STAGES=$total_stages
}

# Render a simple text progress bar
render_bar() {
  local percent="$1" # integer 0..100
  local width=30
  local filled=$(( (percent * width) / 100 ))
  local bar=""
  for ((i=0; i<filled; i++)); do bar+="#"; done
  for ((i=filled; i<width; i++)); do bar+="-"; done
  echo "[$bar] ${percent}%"
}

print_queue() {
  local _unused_idx="$1"  # preserved for compatibility
  echo ""
  # Progress header
  compute_progress
  local percent=0
  if [ "${PROG_TOTAL_STAGES:-0}" -gt 0 ]; then
    percent=$(( (PROG_DONE_STAGES * 100) / PROG_TOTAL_STAGES ))
  fi
  local bar
  bar=$(render_bar "$percent")
  echo "Progress: $bar (${PROG_DONE_STAGES}/${PROG_TOTAL_STAGES})"
  
  # Active variant mini progress
  local active_coll=""
  local active_stage=""
  local Q_NAME Q_DIR Q_VAR Q_SIZE Q_OVERLAP Q_BACKEND
  while IFS='|' read -r Q_NAME Q_DIR Q_VAR Q_SIZE Q_OVERLAP Q_BACKEND; do
    [ -z "$Q_NAME" ] && continue
    local COLL="${Q_NAME}_${Q_VAR}"
    if [ -f "$LOG_DIR/${COLL}.running" ]; then
      active_coll="$COLL"
      active_stage=$(cat "$LOG_DIR/${COLL}.stage" 2>/dev/null || echo "")
      break
    fi
  done < "$LOG_DIR/queue.txt"
  if [ -n "$active_coll" ]; then
    local aidx
    aidx=$(stage_index "$active_stage")
    local aval=$(( (aidx * 100) / 7 ))
    local abar
    abar=$(render_bar "$aval")
    # Timing estimates
    local elapsed; elapsed=$(stage_elapsed_secs "$active_coll" "$active_stage")
    avg_stage_duration "$active_stage"
    local est="-" eta="-"
    if [ "$AVG_STG_SECS" -gt 0 ] 2>/dev/null; then
      est=$(fmt_dur "$AVG_STG_SECS")
      local rem=$(( AVG_STG_SECS - elapsed ))
      if [ "$rem" -lt 0 ]; then rem=0; fi
      eta=$(fmt_dur "$rem")
    fi
    echo "Active: ${active_coll} — ${active_stage} ${abar} | elapsed $(fmt_dur "$elapsed") | est ${est} | eta ${eta}"
  fi
  
  echo "Queue:"
  local idx=0
  local Q_NAME Q_DIR Q_VAR Q_SIZE Q_OVERLAP Q_BACKEND
  while IFS='|' read -r Q_NAME Q_DIR Q_VAR Q_SIZE Q_OVERLAP Q_BACKEND; do
    [ -z "$Q_NAME" ] && continue
    local COLL="${Q_NAME}_${Q_VAR}"
    local status="PENDING"
    local stage=""
    if [ -f "$LOG_DIR/${COLL}.done" ]; then
      status="DONE"
    elif [ -f "$LOG_DIR/${COLL}.running" ]; then
      status="RUNNING"
      stage=$(cat "$LOG_DIR/${COLL}.stage" 2>/dev/null || echo "")
      [ -n "$stage" ] && status="RUNNING: $stage"
    fi
    printf "  [%s] %s :: %s\n" "$status" "$Q_NAME" "$Q_VAR"
    idx=$((idx+1))
  done < "$LOG_DIR/queue.txt"
  echo ""
}

# UI refresh helpers
render_ui_once() {
  printf "\033[2J\033[H"
  echo "E1 Triad Orchestrator (live)"
  echo "Logs: $LOG_DIR"
  print_queue 0
}

ui_loop() {
  while true; do
    render_ui_once
    sleep 2
  done
}

monitor_stats() {
  # monitor_stats <DB> <COLL> <OUTLOG>
  local DB="$1"; local COLL="$2"; local OUTLOG="$3"
  while true; do
    # Lightweight counts via sqlite to avoid heavy CLI/printing
    local counts
    counts=$(bash -lc "$CONDA_PREFIX_CMD >/dev/null 2>&1; python - "$DB" "$COLL" <<'PY'
import sqlite3, sys
db, coll = sys.argv[1], sys.argv[2]
try:
    con = sqlite3.connect(db, timeout=1)
    cur = con.cursor()
    def count(table):
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE collection_id=?", (coll,))
            return cur.fetchone()[0]
        except Exception:
            return -1
    docs = count('documents')
    chunks = count('chunks')
    print(f"{docs} {chunks}")
except Exception:
    print("-1 -1")
PY
    ")
    local docs chunks
    docs=$(echo "$counts" | awk '{print $1}')
    chunks=$(echo "$counts" | awk '{print $2}')
    echo "[$(date '+%H:%M:%S')] Stats — Docs: ${docs}  Chunks: ${chunks}" >> "$OUTLOG"
    sleep "$MONITOR_INTERVAL"
  done
}

run_variant() {
  local NAME="$1"       # scifact | fiqa
  local DS_DIR="$2"     # dataset dir path
  local VAR="$3"        # variant id suffix e.g., fixed_256_20
  local CHUNK_SIZE="$4"
  local CHUNK_OVERLAP="$5"
  local BACKEND="$6"    # token | semantic

  local COLL="${NAME}_${VAR}"
  local DB="data/rag_vectors__${COLL}.db"
  local DOCS_DIR="${DS_DIR}/docs"
  local OUT_DIR="results/e1_${NAME}/full/${COLL}"
  local VLOG="$LOG_DIR/${COLL}.log"

  # Mark running and initial stage
  : > "$LOG_DIR/${COLL}.running"
  start_stage "$COLL" "INGEST"

  log "==> [${NAME}/${VAR}] Ingest (no-embed)"
  ( bash -lc "$CONDA_PREFIX_CMD; ulimit -n 32768 || true; python main.py --db-path '$DB' ingest directory '$DOCS_DIR' \
    --collection '$COLL' --chunk-size '$CHUNK_SIZE' --chunk-overlap '$CHUNK_OVERLAP' \
    --chunk-backend '$BACKEND' --no-embed --deduplicate --resume --max-workers ${INGEST_WORKERS}" >> "$VLOG" 2>&1 ) &
  local INGEST_PID=$!
  monitor_stats "$DB" "$COLL" "$VLOG" &
  local MON_PID=$!
  wait "$INGEST_PID" || { kill "$MON_PID" 2>/dev/null || true; return 1; }
  kill "$MON_PID" 2>/dev/null || true
  end_stage "$COLL" "INGEST"

  # Validate ingest coverage; retry once with safer settings if too low
  local expected_docs actual_docs
  expected_docs=$(find "$DOCS_DIR" -type f -name "*.txt" | wc -l | tr -d ' ')
  actual_docs=$(bash -lc "$CONDA_PREFIX_CMD; python main.py --db-path '$DB' analytics stats --collection '$COLL' 2>/dev/null" \
    | { rg -o "Documents: [0-9]+" -n -S 2>/dev/null || grep -Eo "Documents: [0-9]+" || true; } \
    | head -n1 | awk '{print $2}')
  if [ -z "${actual_docs:-}" ]; then actual_docs=0; fi
  if [ "$expected_docs" -gt 0 ] && [ "$actual_docs" -lt $(( expected_docs * 90 / 100 )) ]; then
    log "Low ingest coverage (${actual_docs}/${expected_docs}). Retrying with max-workers=1 and higher ulimit."
    start_stage "$COLL" "INGEST-RETRY"
    ( bash -lc "$CONDA_PREFIX_CMD; ulimit -n 32768 || true; python main.py --db-path '$DB' ingest directory '$DOCS_DIR' \
      --collection '$COLL' --chunk-size '$CHUNK_SIZE' --chunk-overlap '$CHUNK_OVERLAP' \
      --chunk-backend '$BACKEND' --no-embed --deduplicate --resume --max-workers 1" 2>&1 \
      | tee -a "$VLOG" ) || true
    end_stage "$COLL" "INGEST-RETRY"
  fi

  start_stage "$COLL" "BM25"
  if [ "$FAST_MODE" = "1" ]; then
    log "[FAST_MODE] Skipping BM25 eval for ${COLL}"
    # still record timing slots for consistency
    start_stage "$COLL" "BM25"; end_stage "$COLL" "BM25"
  else
    log "==> [${NAME}/${VAR}] BM25 eval (pre-embed)"
    bash -lc "$CONDA_PREFIX_CMD; python '$EVAL' --db-path '$DB' --embedding-path '$EMB' \
      --collection '$COLL' --dataset-dir '$DS_DIR' --mode bm25 --kdocs 10 --nchunks 100 --cap 3 \
      --emit_diagnostics --outdir '$OUT_DIR'" >> "$VLOG" 2>&1
  fi
  end_stage "$COLL" "BM25"

  start_stage "$COLL" "REEMBED"
  log "==> [${NAME}/${VAR}] Re-embed"
  ( bash -lc "$CONDA_PREFIX_CMD; python main.py --db-path '$DB' maintenance reindex --operation reembed \
      --collection '$COLL' --embedding-path '$EMB' --batch-size 32" >> "$VLOG" 2>&1 ) &
  local REEMBED_PID=$!
  monitor_stats "$DB" "$COLL" "$VLOG" &
  MON_PID=$!
  wait "$REEMBED_PID" || { kill "$MON_PID" 2>/dev/null || true; return 1; }
  kill "$MON_PID" 2>/dev/null || true
  end_stage "$COLL" "REEMBED"

  start_stage "$COLL" "REBUILD"
  log "==> [${NAME}/${VAR}] Rebuild + Vacuum"
  bash -lc "$CONDA_PREFIX_CMD; python main.py --db-path '$DB' maintenance reindex --operation rebuild" >> "$VLOG" 2>&1
  end_stage "$COLL" "REBUILD"
  if [ "$FAST_MODE" = "1" ]; then
    log "[FAST_MODE] Deferring VACUUM for ${COLL}"
    start_stage "$COLL" "VACUUM"; end_stage "$COLL" "VACUUM"
  else
    start_stage "$COLL" "VACUUM"
    bash -lc "$CONDA_PREFIX_CMD; python main.py --db-path '$DB' maintenance reindex --operation vacuum" >> "$VLOG" 2>&1
    end_stage "$COLL" "VACUUM"
  fi

  start_stage "$COLL" "DENSE"
  log "==> [${NAME}/${VAR}] Dense eval"
  bash -lc "$CONDA_PREFIX_CMD; python '$EVAL' --db-path '$DB' --embedding-path '$EMB' \
    --collection '$COLL' --dataset-dir '$DS_DIR' --mode dense --kdocs 10 --nchunks 100 --cap 3 \
    --emit_diagnostics --outdir '$OUT_DIR'" >> "$VLOG" 2>&1
  end_stage "$COLL" "DENSE"

  start_stage "$COLL" "HYBRID"
  log "==> [${NAME}/${VAR}] Hybrid eval"
  bash -lc "$CONDA_PREFIX_CMD; python '$EVAL' --db-path '$DB' --embedding-path '$EMB' \
    --collection '$COLL' --dataset-dir '$DS_DIR' --mode hybrid --kdocs 10 --nchunks 100 --cap 3 \
    --emit_diagnostics --outdir '$OUT_DIR'" >> "$VLOG" 2>&1
  end_stage "$COLL" "HYBRID"
  # Mark done
  rm -f "$LOG_DIR/${COLL}.running" 2>/dev/null || true
  : > "$LOG_DIR/${COLL}.done"
  echo "DONE" > "$LOG_DIR/${COLL}.stage"
}

build_queue() {
  if [ "$FAST_MODE" = "1" ]; then
    cat >"$LOG_DIR/queue.txt" <<EOF
scifact|$SCIDIR|fixed_512_50|512|50|token
fiqa|$FIQDIR|fixed_512_50|512|50|token
EOF
  else
    cat >"$LOG_DIR/queue.txt" <<EOF
scifact|$SCIDIR|fixed_256_20|256|20|token
scifact|$SCIDIR|fixed_512_50|512|50|token
scifact|$SCIDIR|semantic_256_0|256|0|semantic
fiqa|$FIQDIR|fixed_256_20|256|20|token
fiqa|$FIQDIR|fixed_512_50|512|50|token
fiqa|$FIQDIR|semantic_256_0|256|0|semantic
EOF
  fi
}

main() {
  build_queue
  log "Starting E1 triad run (logs: $LOG_DIR)"
  # Start UI loop
  ui_loop &
  UI_PID=$!
  # Ensure UI stops on exit
  trap 'kill "$UI_PID" 2>/dev/null || true' EXIT INT TERM

  local idx=0
  while IFS= read -r line || [ -n "$line" ]; do
    idx=$((idx+1))
    [ -z "$line" ] && continue

    parse_line "$line"

    if [ -z "${NAME:-}" ] || [ -z "${VAR:-}" ] || [ -z "${CHUNK_SIZE:-}" ] || [ -z "${CHUNK_OVERLAP:-}" ] || [ -z "${BACKEND:-}" ]; then
      log "Skipping malformed queue entry: $line"
      log "Parsed fields => NAME='${NAME:-}' VAR='${VAR:-}' SIZE='${CHUNK_SIZE:-}' OVERLAP='${CHUNK_OVERLAP:-}' BACKEND='${BACKEND:-}'"
      continue
    fi

    log "Dispatch => NAME='$NAME' DS_DIR='$DS_DIR' VAR='$VAR' SIZE='$CHUNK_SIZE' OVERLAP='$CHUNK_OVERLAP' BACKEND='$BACKEND'"
    run_variant "$NAME" "$DS_DIR" "$VAR" "$CHUNK_SIZE" "$CHUNK_OVERLAP" "$BACKEND"
  done < "$LOG_DIR/queue.txt"

  log "All variants completed."
  # Auto-generate triad summary after runs complete
  bash -lc "$CONDA_PREFIX_CMD; python 'experiments/gpt-5-pro proposals/scripts/e1_analyze_triad.py' --outdir 'results/e1_triad/summary'" >> "$LOG_DIR/runner.log" 2>&1 || true
  kill "$UI_PID" 2>/dev/null || true
}

main "$@"
