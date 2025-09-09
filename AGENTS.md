# Repository Guidelines

## Python Environment
- Always run Python using the project conda environment:
  `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python`
  Use this prefix for all commands shown below (tests, CLI, scripts).

## Project Structure & Module Organization
- `src/`: Core Python modules (e.g., `rag_pipeline.py`, `vector_database.py`, `embedding_service.py`, `corpus_*`, `deduplication.py`).
- `tests/`: Unit/integration tests (`test_*.py`). Uses `unittest` and lightweight scripts.
- `docs/` and `documentation/`: Architecture and component docs; start with `docs/README.md`.
- `scripts/`: Utilities and test runners (e.g., `scripts/tests/run_retrieval_tests.py`).
- `vendor/sqlite-vec/`: Bundled sqlite-vec extension binaries/headers.
- `config/`: App/model configs (`app_config.yaml`, `model_config.yaml`).
- `examples/`, `benchmarks/`, `sample_corpus/`, `data/`, `models/`, `logs/`, `reports/`.

## Build, Test, and Development Commands
- Run status: `python main.py status`
- Dry-run ingestion: `python main.py ingest directory sample_corpus --collection demo --dry-run`
- Query once: `python main.py query "What is machine learning?" --collection demo`
- All tests: `python -m unittest discover -s tests -v`
- Targeted tests: `python tests/test_vector_database_fix.py` or `python tests/test_extension.py`
- Retrieval suite: `python scripts/tests/run_retrieval_tests.py --config tests/retrieval_test_prompts.json --output test_results`

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation; keep lines readable.
- Naming: modules/functions `snake_case`; classes `CamelCase`; constants `UPPER_SNAKE_CASE`.
- Use type hints and docstrings for public APIs; prefer explicit returns.
- Keep CLI additions in `main.py` (Click commands) and log to `logs/` via the existing helpers.

## Testing Guidelines
- Framework: `unittest` with mocks; avoid loading real models. Patch `SentenceTransformer` and `Llama` (see existing tests).
- Name tests `test_*.py` under `tests/`; use `tempfile` DBs and cleanup.
- For vector DB, tests should work without sqlite-vec but prefer the extension; validate via `tests/test_extension.py`.

## Commit & Pull Request Guidelines
- Commits: imperative mood with optional scope prefix, e.g., `ingest: add resume flag`, similar to `baseline: ...` in history. Keep changes focused.
- PRs: clear description, linked issues, before/after behavior, sample CLI output, and updated tests/docs. Note config changes (DB/model paths) and include reproducible commands.

## Security & Configuration Tips
- Default paths live in `main.py`; override via CLI flags. Some tools respect `LLM_MODEL_PATH`.
- Do not commit large models or private data. Verify sqlite-vec placement in `vendor/sqlite-vec/` and test loading before merging.

## Reranking Experiment Handoff (BEIR v2.2)
- Branch: `feat/experiment-v2-2-reranking-beir`
- Start here: `experiments/reranking/HANDOFF_RERANKING.md` (quick commands) and `experiments/Opus_proposals_v2/v2_2_reranking_BEIR.md` (Section 11: current results + next steps).
- Env for all commands: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`
- DB: `data/rag_vectors.db`; collections: `fiqa_technical`, `scifact_scientific`.
- Runner: `experiments/reranking/run_reranking_experiment.py` (supports `--retrieval-method`, `--alpha`, `--candidate-multiplier`, `--reranker-model`, `--rerank-topk`).
- Evaluator: `experiments/reranking/evaluate_reranking.py` (maps BEIR corpus-ids from filename/source stem).
- Results live under `experiments/reranking/results/`; commit small JSONs only.
