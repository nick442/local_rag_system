# Handoff: Merge Relevant Hybrid Optimization Changes into `main`

## Objective
Merge only the relevant hybrid optimization changes into the `main` branch, using this curated aggregation as the single source of truth. This reduces noise and avoids bringing over transient artifacts or bulky results.

## Curated Contents (Source → Target)
See MANIFEST.json for a machine‑readable list. Summary:

- Top‑level
  - `experiments/hybrid-opus-codex/main.py` → `main.py`
- Core modules
  - `src/retriever.py` → `src/retriever.py`
  - `src/rag_pipeline.py` → `src/rag_pipeline.py`
  - `src/experiment_runner.py` → `src/experiment_runner.py`
  - `src/vector_database.py` → `src/vector_database.py`
- Experiment scripts (kept minimal)
  - `scripts/analyze_results.py` → `experiments/hybrid/analyze_results.py`
  - `scripts/visualize_results.py` → `experiments/hybrid/visualize_results.py`
  - `scripts/monitor_and_analyze.py` → `experiments/hybrid/monitor_and_analyze.py`
- Tools (evaluation + reporting)
  - `tools/augment_results_with_quality.py` → `experiments/hybrid/tools/augment_results_with_quality.py`
  - `tools/prepare_beir_queries_local.py` → `experiments/hybrid/tools/prepare_beir_queries_local.py`
  - `tools/make_query_subset_from_qrels.py` → `experiments/hybrid/tools/make_query_subset_from_qrels.py`
  - `tools/make_query_subset.py` → `experiments/hybrid/tools/make_query_subset.py`
  - `tools/make_exec_summary.py` → `experiments/hybrid/tools/make_exec_summary.py`
- Docs
  - `docs/FINAL_REPORT_v2.md` → `experiments/hybrid/FINAL_REPORT_v2.md`
  - `docs/EXEC_SUMMARY.md` → `experiments/hybrid/EXEC_SUMMARY.md`
  - `docs/EXPERIMENT_REDO_LOG.md` → `experiments/hybrid/EXPERIMENT_REDO_LOG.md`
  - `docs/README_hybrid_experiment.md` → `experiments/hybrid/README.md`
- Config
  - `config/rag_config.yaml` → `experiments/hybrid/config/rag_config.yaml`

## Rationale (Why these files)
- `main.py`: adds explicit `--fusion`, `--cand-mult`, `--rrf-k` sweep flags and saves fusion provenance.
- `src/*`: minimal surgical changes to wire alpha, candidate pooling, fusion methods, and retrieval‑only mode.
- `experiments/hybrid/*`: keep only scripts and tools essential for quality analysis and reporting; exclude large raw results and intermediate logs from merge.
- Docs provide a complete academic‑style record without bloating the repo with bulky artifacts.

## Explicit Exclusions
- Do NOT merge `experiments/hybrid/results/*`, `experiments/hybrid/figures/*`, `experiments/hybrid/analysis/*` unless explicitly requested.
- Do NOT merge large model binaries or environment caches (`models/*`, `__pycache__`, `.git`).
- Do NOT merge local test_data except qrels/queries if explicitly needed for CI (not included here).

## Merge Procedure (Suggested)
1) Create a working branch:
   ```bash
   git checkout main && git pull
   git checkout -b feat/hybrid-fusion-alpha
   ```

2) Copy curated files from this aggregation to their targets:
   ```bash
   AGG=experiments/hybrid-opus-codex
   # Top-level and src
   cp -f $AGG/main.py .
   cp -f $AGG/src/*.py src/
   # Experiments scripts
   mkdir -p experiments/hybrid/tools experiments/hybrid
   cp -f $AGG/scripts/analyze_results.py experiments/hybrid/
   cp -f $AGG/scripts/visualize_results.py experiments/hybrid/
   cp -f $AGG/scripts/monitor_and_analyze.py experiments/hybrid/
   cp -f $AGG/tools/*.py experiments/hybrid/tools/
   # Docs + config
   cp -f $AGG/docs/FINAL_REPORT_v2.md experiments/hybrid/
   cp -f $AGG/docs/EXEC_SUMMARY.md experiments/hybrid/
   cp -f $AGG/docs/EXPERIMENT_REDO_LOG.md experiments/hybrid/
   cp -f $AGG/docs/README_hybrid_experiment.md experiments/hybrid/README.md
   mkdir -p experiments/hybrid/config
   cp -f $AGG/config/rag_config.yaml experiments/hybrid/config/
   ```

3) Review changes (surgical diff):
   ```bash
   git status
   git diff --stat
   ```

4) Commit:
   ```bash
   git add main.py src/*.py experiments/hybrid/*.py experiments/hybrid/tools/*.py \
           experiments/hybrid/*.md experiments/hybrid/config/rag_config.yaml
   git commit -m "hybrid: add fusion options, alpha wiring, tools, and docs"
   ```

5) Push and open PR:
   ```bash
   git push -u origin feat/hybrid-fusion-alpha
   ```

## Verification Checklist
- CLI:
  - `python main.py experiment sweep --help` shows `--fusion`, `--cand-mult`, `--rrf-k`.
- Retrieval-only sweep sanity:
  - `RAG_SWEEP_NO_LLM=1 python main.py --db-path data/rag_vectors.db experiment sweep \
     --param similarity_threshold --values "0.0,0.5,1.0" --queries test_data/fiqa_queries_test_all.json \
     --corpus fiqa_technical --fusion zscore --cand-mult 5 \
     --output experiments/hybrid/results/fiqa_alpha_test_sanity.json`
- Quality augmentation (optional):
  - `python experiments/hybrid/tools/augment_results_with_quality.py --dataset fiqa \
     --results experiments/hybrid/results/fiqa_alpha_test_sanity.json --db data/rag_vectors.db \
     --out experiments/hybrid/results/fiqa_alpha_test_sanity.quality.json`

## Rollback Plan
- If any issues arise, revert with:
  ```bash
  git reset --hard HEAD~1
  git checkout main && git branch -D feat/hybrid-fusion-alpha
  ```

## Notes
- MANIFEST.json enumerates file moves for auditability.
- Keep result artifacts out of the merge to avoid churn and repository bloat.
- The DB (`data/rag_vectors.db`) is not moved; it’s assumed present in the environment.
