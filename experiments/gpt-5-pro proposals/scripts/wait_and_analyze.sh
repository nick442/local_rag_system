#!/usr/bin/env bash
set -euo pipefail

# Waits for two eval waiter PIDs to finish, then runs e1_analyze.py to produce summary CSV/JSON.

# Args:
#  -r <repo_root>
#  -p256 <pid_file_for_256_waiter>
#  -p512 <pid_file_for_512_waiter>
#  -o <outdir>

REPO_ROOT=""
PID256=""
PID512=""
OUTDIR=""

usage(){ echo "Usage: $0 -r repo -p256 pidfile -p512 pidfile -o outdir" >&2; }

while getopts "r:p256:p512:o:" opt; do
  case "$opt" in
    r) REPO_ROOT="$OPTARG";;
    p256) PID256="$OPTARG";;
    p512) PID512="$OPTARG";;
    o) OUTDIR="$OPTARG";;
    *) usage; exit 2;;
  esac
done

if [[ -z "$REPO_ROOT" || -z "$PID256" || -z "$PID512" || -z "$OUTDIR" ]]; then
  usage; exit 2
fi

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
echo "[$(ts)] Waiting for eval waiters to finish..."

wait_on(){ local f="$1"; if [[ -f "$f" ]]; then local p=$(cat "$f" 2>/dev/null||true); if [[ -n "$p" ]]; then while ps -p "$p" >/dev/null 2>&1; do sleep 30; done; fi; fi; }

wait_on "$PID256"
wait_on "$PID512"

echo "[$(ts)] Both eval waiters finished. Running analysis..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rag_env
cd "$REPO_ROOT"
python "experiments/gpt-5-pro proposals/scripts/e1_analyze.py" --outdir "$OUTDIR"
echo "[$(ts)] Analysis complete."
exit 0

