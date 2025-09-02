#!/usr/bin/env python3
"""
Benchmark Runner Entrypoint

Provides minimal classes expected by tests:
- BenchmarkConfig: simple configuration container
- BenchmarkRunner: convenience wrapper to run performance/accuracy suites

This script integrates with the existing benchmark suites under `benchmarks/`.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class BenchmarkConfig:
    """Configuration for running benchmarks."""
    config_path: Optional[str] = None
    output_dir: str = "test_results"
    iterations: int = 5
    queries: Optional[List[str]] = None


class BenchmarkRunner:
    """Run available benchmark suites with a unified interface."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

    def run_performance(self) -> Dict[str, Any]:
        """Run the performance benchmark suite and return a summary dict."""
        from benchmarks.performance_suite import PerformanceBenchmark

        bench = PerformanceBenchmark(config_path=self.config.config_path)
        # Execute a minimal subset to keep CI light
        result = bench.benchmark_token_throughput(
            num_iterations=max(1, self.config.iterations),
            test_queries=self.config.queries,
        )
        return {
            "suite": "performance",
            "name": result.name,
            "duration": result.duration,
            "metrics": result.metrics,
        }

    def run_accuracy(self) -> Dict[str, Any]:
        """Run the accuracy benchmark suite and return a summary dict."""
        from benchmarks.accuracy_suite import AccuracyBenchmark, EvaluationQuery

        bench = AccuracyBenchmark(config_path=self.config.config_path)
        # Minimal sample queries; callers can override via config.queries
        queries = self.config.queries or [
            "What is machine learning?",
            "Explain supervised vs unsupervised learning",
        ]
        eval_queries = [EvaluationQuery(query=q) for q in queries]
        result = bench.evaluate_retrieval_relevance(eval_queries, k_values=[1, 5])
        return {
            "suite": "accuracy",
            "name": result.name,
            "metrics": result.metrics,
        }


if __name__ == "__main__":
    # Optional simple CLI for manual runs
    import argparse, json
    parser = argparse.ArgumentParser(description="Run RAG benchmarks")
    parser.add_argument("suite", choices=["performance", "accuracy"], help="Benchmark suite to run")
    parser.add_argument("--config-path", dest="config_path", default=None)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    runner = BenchmarkRunner(BenchmarkConfig(config_path=args.config_path, iterations=args.iterations))
    out = runner.run_performance() if args.suite == "performance" else runner.run_accuracy()
    print(json.dumps(out, indent=2))

