Continuous Integration

- Skip CI: Include "[skip ci]" or "[ci skip]" in a commit message to skip GitHub Actions workflows for that commit (push and PR events).
- Retrieval Smoke: Minimal workflow runs with mock LLM and mock embeddings; artifacts include JSON results and metrics.jsonl.
- Local Smoke: Use `scripts/tests/run_retrieval.sh --metrics` with `LLM_MODEL_PATH` and `EMBEDDING_MODEL_PATH` to run with local models.

