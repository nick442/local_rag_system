#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper to ingest a small corpus and run the retrieval suite
# Uses the project conda env and attempts to locate local LLM/embedding models

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Activate project conda env
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate rag_env || true
fi

# Resolve LLM model path
pick_llm() {
  local candidates=(
    "$LLM_MODEL_PATH"
    "models/gemma-3-4b-it-q4_0.gguf"
    "/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/models/gemma-3-4b-it-q4_0.gguf"
  )
  for p in "${candidates[@]}"; do
    if [ -n "${p:-}" ] && [ -f "$p" ]; then
      echo "$p"; return 0
    fi
  done
  return 1
}

# Resolve embedding model path (directory with config.json inside)
pick_embeddings() {
  # If env provided and contains config.json, use it
  if [ -n "${EMBEDDING_MODEL_PATH:-}" ] && [ -f "$EMBEDDING_MODEL_PATH/config.json" ]; then
    echo "$EMBEDDING_MODEL_PATH"; return 0
  fi
  # Try local snapshots
  local base1="models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
  if [ -d "$base1" ]; then
    local first
    first="$(ls -1d "$base1"/* 2>/dev/null | head -n1 || true)"
    if [ -n "$first" ] && [ -f "$first/config.json" ]; then
      echo "$first"; return 0
    fi
  fi
  # Try known absolute path from base repo
  local base2="/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/local_rag_system/models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
  if [ -d "$base2" ]; then
    local first2
    first2="$(ls -1d "$base2"/* 2>/dev/null | head -n1 || true)"
    if [ -n "$first2" ] && [ -f "$first2/config.json" ]; then
      echo "$first2"; return 0
    fi
  fi
  return 1
}

LLM_PATH="$(pick_llm || true)"
EMB_PATH="$(pick_embeddings || true)"

if [ -z "$LLM_PATH" ]; then
  echo "Error: Could not locate LLM GGUF model. Set LLM_MODEL_PATH or place model under models/." >&2
  exit 1
fi
if [ -z "$EMB_PATH" ]; then
  echo "Error: Could not locate sentence-transformers embedding snapshot. Set EMBEDDING_MODEL_PATH or place under models/embeddings/." >&2
  exit 1
fi

export LLM_MODEL_PATH="$LLM_PATH"
export EMBEDDING_MODEL_PATH="$EMB_PATH"

echo "Using LLM: $LLM_MODEL_PATH"
echo "Using Embeddings: $EMBEDDING_MODEL_PATH"

# Ensure DB has content by ingesting a tiny sample
echo "Ingesting sample_corpus into collection 'demo'..."
python main.py ingest directory sample_corpus --collection demo --dry-run || true
python main.py ingest directory sample_corpus --collection demo || true

# Run retrieval suite (pass through any CLI args)
OUT_DIR="test_results"
mkdir -p "$OUT_DIR"
python -m scripts.tests.run_retrieval_tests_fixed --config tests/retrieval_test_prompts.json --output "$OUT_DIR" "$@"

echo "Done. Reports in $OUT_DIR"

