# Configuration

Configuration is centralized and validated by `src/config_manager.py`. Values merge from three layers (highest wins):
1) CLI flags (from `main.py`)
2) Environment variables
3) YAML files under `config/`

## Files
- `config/app_config.yaml`: app paths (DB, logs), retrieval defaults, performance knobs.
- `config/model_config.yaml`: model paths and generation defaults.

Common overrides:
- `LLM_MODEL_PATH`: path to local GGUF (overrides model path).
- `RAG_ENABLE_METRICS=1`: enable JSONL metrics collection.
- `RAG_METRICS_PATH=logs/metrics.jsonl`: metrics file path.

## Lookup and Propagation
Use `ConfigManager.get_param(key, default)` to read settings. The manager builds an effective profile (`ProfileConfig`) used by the pipeline and experiment runner so that chunking, retrieval, and generation parameters stay consistent across components.

Priority rules:
- CLI > ENV > YAML defaults.
- Paths are resolved relative to the repo unless absolute.

## Typical Parameters
- Database: `database.path` (e.g., `data/rag_vectors.db`)
- Retrieval: `retrieval_k`, `method` (vector|keyword|hybrid)
- Chunking: `chunk_size`, `chunk_overlap`
- LLM: `llm_model_path`, `max_tokens`, `temperature`, `n_ctx`

## Usage
- CLI entry points in `main.py` initialize `ConfigManager` once and pass the effective profile to `RAGPipeline` and tools.
- `ExperimentRunner` reads base parameters from `ConfigManager` and applies experiment deltas.

## Best Practices
- Keep YAML small and explicit; prefer CLI flags for experiments.
- Never hard‑code paths in modules—fetch via `ConfigManager`.
- Validate dimensions when swapping embedding models; the vector index is created with the actual embedding dimension.

