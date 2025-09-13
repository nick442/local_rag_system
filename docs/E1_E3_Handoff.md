# E1/E3 Handoff – Status, Context, and Next Actions

This handoff summarizes everything needed to continue the E1 (Chunking) and E3 (Hybrid Fusion) experiments, what’s already completed, what is currently running, and the exact commands and files involved.

## Original Experiment Description and References
- Research plan (E1 and broader program): `experiments/gpt-5-pro proposals/RAG_Performance_Research_Plan.md`
- E1 operational handoff: `experiments/gpt-5-pro proposals/E1_Operational_Handoff.md`
- E1 final report (prior): `reports/E1_Chunking_Final_Report.md`
- E1 execution plan (detailed steps): `experiments/gpt-5-pro proposals/E1_Chunking_Execution_Plan.md`

## Environment & Conventions
- Always use the project conda env for Python:
  - `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`
- Root repo paths used below assume current CWD is the repo root.
- Default embedding path in this project:
  - `models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`

## Dataset Layouts
- TREC‑COVID: `experiments/gpt-5-pro proposals/datasets/trec_covid/`
- SciFact: `experiments/gpt-5-pro proposals/datasets/scifact/` (converted from ZIP)
- FiQA‑2018: `experiments/gpt-5-pro proposals/datasets/fiqa/` (converted from ZIP)
- Each dataset dir contains: `docs/*.txt`, `queries.jsonl`, `qrels.tsv`.

## Scripts Added/Modified (key points)
- E1 evaluator (doc‑aware ranking + diagnostics):
  - `experiments/gpt-5-pro proposals/scripts/e1_eval.py`
  - Adds `--emit_diagnostics`: writes `latency*.json`, `position*.json`, and `per_query*.jsonl` (per‑query ranked docs + best‑chunk diagnostics).
  - More robust qrels loader (accepts 3‑ or 4‑column tsv; ignores headers).
- E1 analysis (TREC‑COVID summary):
  - `experiments/gpt-5-pro proposals/scripts/e1_analyze.py`
- Orchestrator with live queue + timing + progress bar:
  - `experiments/gpt-5-pro proposals/scripts/run_e1_triad.sh`
  - Features: two‑phase ingest (no‑embed → reembed → rebuild → vacuum), diagnostics, live UI refresh, per‑variant logs, per‑stage timing, fast‑mode toggles.
  - Tunables via env:
    - `FAST_MODE=1` (skip BM25 per variant, defer VACUUM; queue only 512/50 variants)
    - `INGEST_WORKERS` (default 4), `REEMBED_BATCH` (default 64)
    - `OMP_NUM_THREADS` family defaults to 2 (override as needed)
    - Raises file descriptors (`ulimit -n` up to 32768) to avoid EMFILE.
- E3 fusion (RRF vs Z‑score) + analyzer:
  - `experiments/gpt-5-pro proposals/scripts/e3_fusion.py`
    - Reads/writes per‑query files. If missing, it calls `e1_eval.py` to generate them.
    - Aliases internal DB `doc_id` → corpus doc id (filename stem) for qrels alignment.
  - `experiments/gpt-5-pro proposals/scripts/e3_analyze_trec_covid.py` → summary CSV/JSON for TREC‑COVID fusion.
- Fusion watcher (auto‑run E3 when per‑query files appear):
  - `experiments/gpt-5-pro proposals/scripts/watch_and_fuse.sh`

## Results/Logs – Where to Look
- E1 orchestrator logs, markers, and timing:
  - `results/e1_triad/logs/runner.log` (orchestrator)
  - Per‑variant logs: `results/e1_triad/logs/<dataset>_<variant>.log`
  - Stage marker: `results/e1_triad/logs/<dataset>_<variant>.stage`
  - Per‑stage timing: `results/e1_triad/logs/<dataset>_<variant>.timing`
- Dataset outputs (examples):
  - TREC‑COVID (smoke/test): `results/e1_trec_covid/test/trec_covid_<variant>/`
  - SciFact (full): `results/e1_scifact/full/scifact_<variant>/`
  - FiQA (full): `results/e1_fiqa/full/fiqa_<variant>/`
- E3 fusion outputs (same dirs):
  - `metrics_rrf.json`, `metrics_zscore.json`, `per_query_rrf.jsonl`, `per_query_zscore.jsonl`, `FUSION_README.md`

## Current Status (at handoff)
- TREC‑COVID (smoke):
  - E1: Dense/BM25/Hybrid + diagnostics done for `fixed_256_20`, `fixed_512_50`, `semantic_256_0`.
  - E3: Fusion complete for all three. Summary: `results/e3_trec_covid/summary/{summary.csv,summary.json}`.
- SciFact (fast‑mode, 512/50 only):
  - Orchestrator running. Current stage: `REBUILD` (per marker), prior stages done: INGEST, BM25, REEMBED.
  - Outputs will land in: `results/e1_scifact/full/scifact_fixed_512_50/`.
- FiQA (fast‑mode, 512/50 only):
  - Queued; will run after SciFact finishes. Outputs: `results/e1_fiqa/full/fiqa_fixed_512_50/`.
- Fusion watchers (auto E3 on completion):
  - Script: `watch_and_fuse.sh` set up to trigger `e3_fusion.py` when `per_query.jsonl` and `per_query_bm25.jsonl` appear for SciFact/FiQA 512/50.

## Representative Numbers (TREC‑COVID, 50‑query smoke)
- RRF fusion:
  - `fixed_256_20`: NDCG@10 ≈ 0.403, MRR ≈ 0.740
  - `fixed_512_50`: NDCG@10 ≈ 0.475, MRR ≈ 0.751
  - `semantic_256_0`: NDCG@10 ≈ 0.487, MRR ≈ 0.749
- Z‑score closely tracks RRF, slightly lower NDCG/MRR in this smoke run.

## How to Continue
1) Let the orchestrator finish E1 fast‑mode for SciFact/FiQA (512/50):
   - Live UI is handled by `run_e1_triad.sh` (prints a progress bar and queue). If not visible, tail logs:
     - `tail -f results/e1_triad/logs/runner.log`
     - `tail -f results/e1_triad/logs/scifact_fixed_512_50.log`
   - Stage markers: `cat results/e1_triad/logs/scifact_fixed_512_50.stage` (or FiQA).

2) Fusion (E3) for SciFact/FiQA 512/50 will auto‑run via watcher once per‑query files exist.
   - Manual run, if needed:
     - SciFact: `python "experiments/gpt-5-pro proposals/scripts/e3_fusion.py" --dataset-dir "experiments/gpt-5-pro proposals/datasets/scifact" --outdir "results/e1_scifact/full/scifact_fixed_512_50" --db-path "data/rag_vectors__scifact_fixed_512_50.db" --embedding-path "<EMB_PATH>" --collection "scifact_fixed_512_50" --kdocs 10 --nchunks 100 --cap 3 --rrf_k 60`
     - FiQA: `python "experiments/gpt-5-pro proposals/scripts/e3_fusion.py" --dataset-dir "experiments/gpt-5-pro proposals/datasets/fiqa" --outdir "results/e1_fiqa/full/fiqa_fixed_512_50" --db-path "data/rag_vectors__fiqa_fixed_512_50.db" --embedding-path "<EMB_PATH>" --collection "fiqa_fixed_512_50" --kdocs 10 --nchunks 100 --cap 3 --rrf_k 60`

3) Triad summaries
   - E1 (retrieval modes): `python "experiments/gpt-5-pro proposals/scripts/e1_analyze.py" --outdir results/e1_trec_covid/full/summary` (TREC‑COVID only). For a cross‑dataset triad, extend similarly or consolidate outputs under `results/e1_triad/summary/` (the orchestrator auto‑calls this at completion).
   - E3 (fusion) TREC‑COVID already summarized: `results/e3_trec_covid/summary/`.
   - If desired, add a triad fusion analyzer mirroring `e3_analyze_trec_covid.py` once SciFact/FiQA fusion results exist.

4) Position sensitivity
   - Each run writes `position*.json` with `top1_relative_positions` and histogram bins. Roll up across datasets/variants and include a short interpretation in the report.

5) Reporting
   - Include metrics (P@k, NDCG@10, MRR), latency (mean/p50/p95), and position histograms. For E3, highlight that RRF≈Z‑score with slight edge for RRF in smoke runs; confirm on full.

## Useful Commands (verbatim)
- Orchestrator (fast‑mode) example:
  - `FAST_MODE=1 INGEST_WORKERS=8 REEMBED_BATCH=96 OMP_NUM_THREADS=4 bash "experiments/gpt-5-pro proposals/scripts/run_e1_triad.sh"`
- Stop orchestrator cleanly:
  - `kill -INT "$(cat results/e1_triad/run.pid)"`
- Check active stage and timing:
  - `cat results/e1_triad/logs/<dataset>_<variant>.stage`
  - `tail -n 20 results/e1_triad/logs/<dataset>_<variant>.timing`
- TREC‑COVID E3 summary:
  - `python "experiments/gpt-5-pro proposals/scripts/e3_analyze_trec_covid.py" --outdir results/e3_trec_covid/summary`

## Known Issues & Mitigations
- OpenMP/SHM errors on macOS: mitigated by setting threads to 1–4 (`OMP_NUM_THREADS`, etc.). The orchestrator exports safe defaults (2). Adjust upward if stable.
- Too many open files (EMFILE) during ingest: mitigated by raising `ulimit -n` and reducing `INGEST_WORKERS` on retry.
- Noisy progress bars: disabled in runner (`TQDM_DISABLE=1`, etc.). All details go to per‑variant logs.
- Long REBUILD/VACUUM steps: IO‑bound; fast‑mode defers per‑variant VACUUM and can run a final pass if needed.

## What’s Left to “Declare E1 Done”
- Complete SciFact and FiQA 512/50 runs (currently in progress) with dense, hybrid, latency/position diagnostics.
- Ensure E3 fusion (RRF + Z‑score) runs for both datasets (watchers or manual).
- Aggregate triad summary (E1 + E3) and write a short conclusion:
  - Which chunking variant wins per dataset?
  - Do fusion methods improve over dense/hybrid and by how much?
  - Latency vs. NDCG@10 trade‑off snapshot.
  - Position sensitivity notes (e.g., “lost in the middle” tendencies).

## Pointers
- Embedding model path used:
  - `models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`
- Databases created by E1 fast‑mode:
  - `data/rag_vectors__scifact_fixed_512_50.db`
  - `data/rag_vectors__fiqa_fixed_512_50.db`
- Variant output roots:
  - `results/e1_scifact/full/scifact_fixed_512_50/`
  - `results/e1_fiqa/full/fiqa_fixed_512_50/`

If anything stalls, check the variant’s `.log` and `.stage`, and the orchestrator `runner.log`. The system is designed so you can re‑run individual steps (e.g., call `e1_eval.py` or `e3_fusion.py` directly) without impacting other artifacts.
