# Experiment 1 — Critical Review

## Summary
- Goal: Optimize `chunk_size` and `chunk_overlap` for retrieval/generation performance.
- Reported finding: Best at `chunk_size=256`, `chunk_overlap=192` with faster and more consistent response times.
- Reality: The experiment did not manipulate chunking in the retrieval pipeline or database; conclusions about chunking are not supported by the implementation.

## What’s Good
- Clear plan and log: `experiments/experiment_1/plan.md` and `experiment_1/experiment_log.md` are structured and reproducible in spirit.
- Results captured: `baseline_performance.json`, `chunk_size_sweep.json`, `chunk_overlap_sweep.json`, plus readable analyses (`preliminary_analysis.md`, `comprehensive_analysis.md`).
- Hardware/system notes included; runs completed without failures.

## Critical Issues (invalidate conclusions)
- No applied re-chunking per configuration:
  - `ExperimentRunner._create_rag_pipeline` only overrides `retrieval_k`, `max_tokens`, `temperature`; chunking params are ignored.
  - The pipeline queries the existing index; there’s no per-config re-chunking or collection isolation before runs.
- Collection isolation missing end-to-end:
  - `VectorDatabase.search_similar` supports `collection_id`, but `Retriever.retrieve` doesn’t pass it; `RAGPipeline.query` also discards `collection_id` on the retriever call. All configs query the same global data.
- Results lack run-time parameter provenance:
  - JSON exports omit the per-run `config` (e.g., chunk_size/overlap), preventing traceability and post hoc analysis.
- Metrics insufficient to evaluate chunking:
  - Only coarse `response_time`, `num_sources`, `response_length`. No retrieval precision/recall, no prompt/context token counts, no retrieval vs generation timing, no index size/chunk counts per config.
  - Identical response content across configs suggests changes had no effect (consistent with no chunking applied).
- Undersized and biased query set:
  - Only 3 generic queries used; plan called for a broader/labeled set (e.g., 50+). Statistical claims aren’t defensible with n=3 per config.
- Timing fields inconsistent:
  - In the saved JSON, `created_at` and `completed_at` are the same, despite non-zero `total_runtime`.
- Questionable claims in analysis:
  - “11% reduction in memory usage per chunk” is incorrect: embedding vectors are fixed-size; smaller textual chunks increase count, often growing the index.
  - Recommending `chunk_overlap=192` with `chunk_size=256` implies 75% overlap, contradicting the plan’s 10–25% guidance and likely inflating index/storage.

## Evidence Pointers (repo paths)
- Ignored chunking in experiments: `src/experiment_runner.py` (`_create_rag_pipeline`, `_run_single_experiment`) does not apply chunk_size/overlap; only LLM params are overridden.
- Collection filter not used in retrieval: `src/rag_pipeline.py` drops `collection_id` when calling `Retriever`; `src/retriever.py` retrieve methods do not forward `collection_id`; `src/vector_database.py` can filter by collection but isn’t exercised.
- Exports omit config: `main.py` (`_save_experiment_results`) writes runs without `result.config` (no parameter provenance in JSON).
- Templates define chunk params but runner doesn’t apply them: `src/experiment_templates.py` vs runner behavior above.

## Required Fixes (to produce valid chunking results)
1. Materialize chunking per configuration
   - Before querying, re-chunk and (optionally) re-embed the target corpus with the test config using `ReindexTool.rechunk_documents()`.
   - Isolate data by configuration: clone or derive a per-config collection (e.g., `prod_cs256_co64`), re-chunk there, then run queries against that collection.
2. Thread `collection_id` through the stack
   - `RAGPipeline.query` should pass `collection_id` to `Retriever`.
   - `Retriever.retrieve` should forward `collection_id` into `VectorDatabase.search_similar(..., collection_id=...)` and keyword/hybrid paths.
3. Log parameters and richer metrics
   - Include `result.config` in exports (chunk_size/overlap, retrieval_k, profile, `collection_id`, model paths, commit hash).
   - Capture retrieval vs generation time, prompt/context token counts, number of chunks in collection, index size on disk.
4. Use a proper evaluation set and statistics
   - Run with ≥50 labeled queries; compute R@k, MRR, precision/recall.
   - Use paired analysis per query across configs, ≥10 repetitions per config; report confidence intervals instead of single p-values from n=3.
5. Constrain overlap search space
   - Start with 10–25% of `chunk_size` (per plan); only explore higher overlaps if justified and report index/storage impacts.
6. Repair exports/timing
   - Populate `created_at` and `completed_at` accurately; record run start/end times.

## Suggested Implementation Steps
- Implement `collection_id` threading (pipeline → retriever → vector DB).
- Add a small experiment driver that, per config: (a) create or reuse a per-config collection; (b) re-chunk (and re-embed if needed); (c) run queries; (d) export results with full config and metrics.
- Enhance `_save_experiment_results` to include per-run `config` and additional metrics.

## Suggested Commands (after fixes)
- Re-chunk per config (example):
  - Programmatically via `ReindexTool.rechunk_documents(collection, chunk_size=..., chunk_overlap=..., reembed=true)` into a new collection ID.
- Run sweeps with explicit output and full configs recorded:
  - `python main.py experiment sweep --param chunk_size --values 128,256,512,768,1024 --output results/chunk_size_sweep.json`
  - `python main.py experiment sweep --param chunk_overlap --values 32,64,128,192 --output results/chunk_overlap_sweep.json`
- Evaluate retrieval quality on labeled set:
  - `python scripts/tests/run_retrieval_tests.py --config tests/retrieval_test_prompts.json --output test_results`

## Verdict
As implemented, Experiment 1 does not test document chunking; its conclusions are not supported. With per-config re-chunking, collection isolation, correct parameter threading, richer metrics, and a larger labeled query set, the experiment can yield valid, actionable results. I can implement the wiring (collection_id threading, per-run config export) and a minimal re-chunk-per-config flow to make this experiment valid and reproducible.

