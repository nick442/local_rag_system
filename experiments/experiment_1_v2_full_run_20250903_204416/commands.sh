#!/usr/bin/env bash
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
yes | python main.py experiment template chunk_optimization   --queries test_data/enhanced_evaluation_queries.json   --collection-prefix exp_full_cs   --output "experiments/experiment_1_v2_full_run_20250903_204416/results.json" | tee -a "experiments/experiment_1_v2_full_run_20250903_204416/run.log"
