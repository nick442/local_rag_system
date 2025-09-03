# Experiment 1 v2 — Chunking Optimization (Default Base)

This report documents the end-to-end execution of Experiment 1 v2 on a small default corpus, the code fixes implemented to enable valid chunking experiments, and a comprehensive guide to scale up the run.

## Executive Summary

- Prepared 20 per-config collections (`exp_cs{128,256,512,768,1024}_co{~10%,~15%,~20%,~25%}`) from base collection `default`, re-embedded with MiniLM-L6-v2.
- Verified pipeline wiring and metrics; added missing features and fixes (see Code Fixes).
- Ran two A/B tests comparing `cs=256, co=64` vs `cs=512, co=128` with queries known to match the default corpus. Retrieval returned contexts and generation proceeded with context.
- Preliminary latency indicates `cs=512, co=128` slightly faster end-to-end on this tiny set; not statistically significant with n=3.

Artifacts in this folder:
- `README.md` — Short log and pointers
- `run.log` — Full console output of runs
- `system_info.json` — Environment details
- `collections_summary.csv` — Chunk/document counts per experimental collection
- `ab_results.json` — Initial A/B (not matching corpus; 0 contexts)
- `ab_results_matching.json`, `ab_results_matching_2.json` — A/B with corpus-matching queries; retrieval active
- `config_a.json`, `config_b.json`, `config_a2.json`, `config_b2.json` — A/B configs
- `queries_matching_default.json` — Query set aligned to the default corpus

## Environment & Setup

- Conda env: `rag_env`
- Symlinks created to share data/models from the main worktree:
  - `models -> /Users/nickwiebe/.../local_rag_system/models`
  - `data   -> /Users/nickwiebe/.../local_rag_system/data`
- LLM model: `models/gemma-3-4b-it-q4_0.gguf`
- Embedding model snapshot: `models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed...`
- Vector DB: `data/rag_vectors.db` (sqlite-vec operational)

## Code Fixes Implemented

1) New CLI: clone + rechunk into per-config collections

- File: `main.py`
- Command: `python main.py collection clone <source> --target <exp_cs...> --chunk-size <int> --chunk-overlap <int> [--embedding-path <path>] [--no-embed]`
- Purpose: Create isolated collections per chunking configuration (copy logical content; re-chunk; optionally re-embed).

2) ReindexTool additions and fixes

- File: `src/reindex.py`
- Added `clone_collection_with_chunking(source, target, chunk_size, chunk_overlap, reembed, backup)`:
  - Reconstructs document content from `source_collection` chunks
  - Creates new `target_collection` ids and re-chunks with requested params
  - (Optional) generates embeddings; gracefully skips if model unavailable
- Fixed `reembed_collection` to UPSERT into both `embeddings` and `embeddings_vec`:
  - Replaced `UPDATE` with `INSERT OR REPLACE` for robustness

3) ExperimentRunner improvements

- File: `src/experiment_runner.py`
- Enhanced metrics captured from `RAGPipeline` metadata (retrieval/generation times, token counts, contexts_count)
- Record `collection_id` on `result.config` for export provenance
- Important selection fix: prefer `derived_collection` (from `chunk_size/overlap`) over `target_corpus` labels (avoids accidental selection of `config_A/B` instead of `exp_cs...` collections in A/B runs)

4) EmbeddingService robustness

- File: `src/embedding_service.py`
- Fixed a runtime error (`name 'torch' is not defined`) by importing `torch` lazily and providing a safe CPU-only fallback path
- Hardened cache clearing and progress reporting when CUDA/MPS are not available

## Data Preparation

Prepared 20 per-config experimental collections from base `default`:

```
exp_cs128_co{13,19,26,32}
exp_cs256_co{26,38,51,64}
exp_cs512_co{51,77,102,128}
exp_cs768_co{77,115,154,192}
exp_cs1024_co{102,154,205,256}
```

Re-embedding executed successfully; sample counts (from `collections_summary.csv`) confirm per-config chunk counts.

## Runs & Preliminary Findings

1) Initial A/B (small baseline queries; not corpus-aligned):
- Results: `ab_results.json`
- Retrieval returned 0 contexts for both configs → not a valid chunking comparison
- Timing differences dominated by cold-start overhead; no conclusions

2) A/B with queries matching the default corpus:
- Results: `ab_results_matching.json`, `ab_results_matching_2.json`
- Retrieval active (e.g., 5 contexts retrieved for “What is deep learning?”)
- Latency (n=3):
  - A (cs=256, co=64): ~4.84s
  - B (cs=512, co=128): ~3.86s
  - ≈20% faster for B on this tiny set; not statistically significant (p≈0.36–0.70 across two short runs)
- Notes:
  - Generation dominates end-to-end time; retrieval_time is relatively small
  - Differences here are indicative only; a larger query set is required

## Limitations

- Base corpus `default` is intentionally small; query alignment is required for non-zero contexts.
- n=3 is too small to conclude; use ≥52 queries for significance.
- Current results reflect a single hardware/software environment and a specific LLM/embedding pair.

## Guide: Scaled-Up Experiment (Recommended)

Objective: Run a statistically meaningful experiment over 20 chunking configs with 52 queries, capturing full timing/token metrics and provenance.

### 1. Preconditions

- Ensure symlinks or actual presence of:
  - `models/gemma-3-4b-it-q4_0.gguf`
  - `models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<snapshot>`
  - `data/rag_vectors.db` with base corpus and `sqlite-vec` available
- Conda env: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`

### 2. Prepare per-config collections

Option A (existing default base):

```
for cs in 128 256 512 768 1024; do
  for r in 0.10 0.15 0.20 0.25; do
    co=$(python - <<PY
cs=$cs; r=$r
print(int(cs*r+0.5))
PY
)
    python main.py collection clone default \
      --target exp_cs${cs}_co${co} \
      --chunk-size $cs --chunk-overlap $co \
      --embedding-path models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<snapshot>
  done
done
```

Option B (larger base for stronger signal, e.g., `realistic_full_production`):

```
# Staged batches to avoid long single jobs
python main.py collection clone realistic_full_production --chunk-size 256 --chunk-overlap 64 \
  --target exp_cs256_co64 --embedding-path models/.../<snapshot>
# Repeat for the remaining 19 configs in batches of 3–5
```

After cloning, (re)embed to ensure vector search is available (idempotent):

```
python main.py maintenance reindex --collection exp_cs256_co64 --operation reembed --no-backup
python main.py maintenance reindex --collection exp_cs512_co128 --operation reembed --no-backup
# ... repeat for all exp_cs*_co* collections
```

### 3. Run the template (20 configs × 52 queries)

```
python main.py experiment template chunk_optimization \
  --queries test_data/enhanced_evaluation_queries.json \
  --output experiments/experiment_1_v2_run_default_base_YYYYMMDD_HHMMSS/results.json
```

Notes:
- The template will iterate over the parameter ranges and will use the derived `exp_cs*_co*` collection per config.
- If you are prompted due to high run counts, confirm to proceed (you can prefix with `yes |` for non-interactive runs).

### 4. Post-run Analysis

Quick timing review (already available in results): average response times per config.

Deeper analysis (optional):

```
python - << 'PY'
from src.evaluation_metrics import ExperimentAnalyzer
import json
data = ExperimentAnalyzer().analyze_experiment_results('experiments/.../results.json')
print(json.dumps(data, indent=2))
PY
```

Retrieval quality (optional): integrate relevance ground-truth and compute P@K / NDCG with `RetrievalQualityEvaluator` once retrieval outputs are persisted per run or using the retrieval test harness.

### 5. Resource & Time Considerations

- For the larger base `realistic_full_production`, cloning + re-embedding will take significant time and disk space; run in batches and `maintenance reindex --operation vacuum` periodically.
- Monitor memory; LLM generation dominates latency. Consider profile tweaks if needed.

## Next Steps / Recommendations

1. Run the 52-query template across all 20 configs on the small `default` base to verify the full stack.
2. Repeat on a larger base corpus (e.g., `realistic_full_production`) for a meaningful chunking signal.
3. Add retrieval precision/recall logging per run (extend exporter to persist retrieved contexts IDs) for IR-quality metrics.

## Reproducible Commands (This Run)

See `commands.sh` and `run.log` in this folder. A/B runs executed here:

```
# A/B with matching queries
python main.py experiment compare \
  --config-a experiments/.../config_a2.json \
  --config-b experiments/.../config_b2.json \
  --queries experiments/.../queries_matching_default.json \
  --output experiments/.../ab_results_matching_2.json
```

---

Prepared by the coding agent (experiment rerun and infrastructure fixes).

