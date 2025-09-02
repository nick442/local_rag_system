# Core Architecture

This repository implements a modular Retrieval-Augmented Generation (RAG) system. The core flow is: ingest corpus → build/search indexes → assemble context → generate an answer. Components are small, testable, and wired via a thin pipeline and CLI.

## Components
- Ingestion: `src/document_ingestion.py`, `src/corpus_manager.py`, `src/corpus_organizer.py`, `src/deduplication.py`.
- Embeddings: `src/embedding_service.py` creates and caches embeddings.
- Vector Index: `src/vector_database.py` provides vector, keyword (FTS5), and hybrid search.
- Retrieval: `src/retriever.py` orchestrates query embedding and top‑k search with optional hybrid scoring and context assembly.
- LLM: `src/llm_wrapper.py` wraps local LLMs; paths and params come from config.
- Pipeline: `src/rag_pipeline.py` coordinates retrieval + generation and exposes a simple API used by the CLI.
- Experiments: `src/experiment_runner.py` runs parameter sweeps and A/B tests, persisting results to SQLite.
- Metrics: `src/metrics.py` provides JSONL logging; disabled by default, toggled via env/CLI.

## Data Flow
1. Documents are chunked and stored with metadata.
2. Query is embedded and searched against the index (vector/keyword/hybrid).
3. Top results are deduplicated and trimmed to a token budget, then formatted into a context block.
4. LLM generates an answer using a prompt assembled by `src/prompt_builder.py`.

## Storage
- SQLite database contains chunks, metadata, and (optionally) the `sqlite-vec` extension for fast similarity search.
- Experiments are stored in `data/experiments.db` via `ExperimentDatabase`.
- Logs and metrics live under `logs/`.

## Configuration
Configuration comes from `config/*.yaml`, environment variables, and CLI flags. `src/config_manager.py` centralizes lookup and validation so defaults live in one place and changes propagate consistently.

## Performance
- Token‑budgeted context assembly prevents over‑long prompts.
- Optional PRAGMAs and indexes in `vector_database.py` keep queries snappy.
- `src/model_cache.py` reduces re‑loads for embeddings and LLMs.

## Extensibility
- New retrievers can implement `interfaces/*_interface.py` and be swapped in `retriever.py`.
- Alternate vector backends can be added via `create_vector_index`.
- CLI is centralized in `main.py`; add subcommands without touching core modules.

