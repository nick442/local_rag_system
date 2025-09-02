# Metrics

Lightweight JSONL metrics are available via `src/metrics.py`. Collection is disabled by default and can be enabled at runtime.

## Enable
- Env: `RAG_ENABLE_METRICS=1`
- Optional: `RAG_METRICS_PATH=logs/metrics.jsonl`
- Programmatic: `from src.metrics import enable_metrics; enable_metrics(True, "logs/metrics.jsonl")`

## What’s Tracked
- Component: logical subsystem (e.g., `retriever`, `pipeline`).
- Event: action within the component (e.g., `retrieve`, `query`).
- Data: structured fields such as `k`, `method`, `duration`, `results`, `collection_id`.

Example JSONL line:
```
{"timestamp": 1710000000.0, "component": "retriever", "event": "retrieve", "data": {"method": "vector", "k": 5, "duration": 0.012, "results": 5}}
```

## Usage in Code
- `retriever.retrieve()` wraps calls with `metrics.track(...)` in a `finally` block to ensure timing is recorded.
- `RAGPipeline` can emit timings for end‑to‑end queries.

## Operational Notes
- Append‑only writes; safe across threads/processes.
- If directory creation or writes fail, the collector degrades to no‑op and does not crash the app.
- Keep metrics minimal; use external tools for heavy analysis.

