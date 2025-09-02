# Results

Experimental outputs should be easy to diff, aggregate, and visualize.

## Where Results Go
- Retrieval suite: `--output <dir>` (e.g., `test_results/run_YYYYMMDD_hhmm/`).
- Experiment runner: structured tables in `data/experiments.db` (`experiments`, `experiment_runs`).

## Files
- JSONL: one result per line with prompt, response, and scores.
- CSV: compact summary for spreadsheets.
- HTML: human‑friendly report with tables and highlights.

## Result Schema (JSONL)
```
{
  "id": "smoke_1",
  "query": "…",
  "response": "…",
  "metrics": {"response_time": 0.42, "source_count": 3, "quality": 0.7},
  "retrieved_documents": [{"source": "…", "score": 0.88}],
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Good Practices
- Keep raw JSONL; derive CSV/plots from it.
- Include the exact command line (and git SHA if available) at the top of reports for reproducibility.
- Use a fresh output folder per run; never overwrite prior results.

