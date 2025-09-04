# Experiment 1 v2 — Full Template Run (realistic_full_production)

This run uses the `chunk_optimization` template with the `exp_full_cs` collection prefix to test 20 chunking configurations on the full base corpus.

## Launch Command

```
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
yes | python main.py experiment template chunk_optimization \
  --queries test_data/enhanced_evaluation_queries.json \
  --collection-prefix exp_full_cs \
  --output RESULTS_DIR/results.json | tee -a RESULTS_DIR/run.log
```

Replace `RESULTS_DIR` with this folder path.

## Notes
- Ensure all `exp_full_cs*` collections exist and have embeddings for valid retrieval.
- This run can take many hours; recommend overnight execution.
- Progress and any warnings will appear in `run.log`.
