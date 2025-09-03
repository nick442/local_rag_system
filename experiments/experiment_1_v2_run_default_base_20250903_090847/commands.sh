#!/usr/bin/env bash
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
# Run template with a small baseline query set (for quick completion)
python main.py experiment template chunk_optimization \
  --queries test_data/small_baseline_queries.json \
  --output experiments/experiment_1_v2_run_default_base_20250903_090847/results.json
