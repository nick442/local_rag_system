# Experiments

Two complementary paths exist for experiments:
1) Experiment Runner for parameter sweeps and A/B tests.
2) Retrieval Test Suite for reproducible retrieval checks.

## Experiment Runner
- Module: `src/experiment_runner.py`
- Persists metadata/results to `data/experiments.db`.
- Supports parameter sweeps (`run_parameter_sweep`) and A/B tests (`run_ab_test`).
- Uses `ConfigManager` profiles and applies per‑run deltas (e.g., `retrieval_k`, `chunk_size`, `temperature`).

Creating a runner:
```python
from src.config_manager import ConfigManager, ExperimentConfig, ParameterRange
from src.experiment_runner import ExperimentRunner

cm = ConfigManager()
runner = ExperimentRunner(cm)
base = ExperimentConfig()
ranges = [ParameterRange(name="retrieval_k", values=[3,5,8])]
queries = ["What is RAG?", "Vector search advantages?"]
results = runner.run_parameter_sweep(base, ranges, queries)
```

## Retrieval Test Suite
- Script: `scripts/tests/run_retrieval_tests.py`
- Config: `tests/retrieval_test_prompts.json`
- Output directory contains JSON lines, CSV summary, and an HTML report.

Run:
```bash
python scripts/tests/run_retrieval_tests.py --config tests/retrieval_test_prompts.json --output test_results
```

## Tips
- Keep queries small and targeted; prefer categories to group prompts.
- Use `--output` per run to preserve history.
- Enable metrics (`RAG_ENABLE_METRICS=1`) when comparing end‑to‑end latency.

