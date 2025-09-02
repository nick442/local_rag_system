# Developer Guide

## Environment
- Use the project conda env for all Python commands:
  `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python`

## Key Commands
- Status: `python main.py status`
- Dry‑run ingest: `python main.py ingest directory sample_corpus --collection demo --dry-run`
- Query: `python main.py query "What is machine learning?" --collection demo`
- Tests (all): `python -m unittest discover -s tests -v`
- Retrieval suite: `python scripts/tests/run_retrieval_tests.py --config tests/retrieval_test_prompts.json --output test_results`

## Code Style
- Python, PEP 8, 4‑space indent; type hints on public APIs.
- Modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.

## Project Map
- Pipeline: `src/rag_pipeline.py`
- Retrieval: `src/retriever.py`, `src/vector_database.py`, `src/embedding_service.py`
- LLM wrapper: `src/llm_wrapper.py`
- Experiments: `src/experiment_runner.py`
- Metrics: `src/metrics.py`

## Testing
- `unittest` with mocks; avoid loading real models in unit tests.
- Prefer temp DBs and cleanup; see `tests/*` for patterns.

## Adding Features
- New retriever/backend: implement `interfaces/*_interface.py`, wire via factories, and add tests.
- New CLI: add a `click` command in `main.py`, use existing logging helpers, and document in `docs/`.

## Tips
- Use `src/model_cache.py` to avoid repeated model loads.
- Validate embedding dimensions when swapping models; the DB is created with the effective dimension.
- Keep docs small; link to code where possible.

