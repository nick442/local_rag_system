#!/usr/bin/env python3
"""
Automated Benchmark Execution for RAG System

Automated benchmark runner:
1. Setup test environment
2. Run all benchmarks
3. Generate reports
4. Compare with baselines
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from benchmarks.performance_suite import PerformanceBenchmark, BenchmarkResult
from benchmarks.accuracy_suite import AccuracyBenchmark, EvaluationQuery, AccuracyResult
from test_data.generate_test_corpus import TestCorpusGenerator

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    run_performance: bool = True
    run_accuracy: bool = True
    use_test_corpus: bool = True
    generate_corpus: bool = True
    corpus_size: str = "small"  # small, medium, large, all
    num_performance_iterations: int = 5
    save_detailed_results: bool = True
    compare_with_baseline: bool = True
    output_format: str = "markdown"  # markdown, json, html

class BenchmarkRunner:
    """Automated benchmark runner for RAG system"""
    
    def __init__(self, config: BenchmarkConfig = None):
        """Initialize benchmark runner"""
        self.config = config or BenchmarkConfig()
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.reports_dir = Path("reports/benchmarks")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_data_dir = Path("test_data")
        self.baseline_file = self.reports_dir / "baseline_metrics.json"
        
        # Initialize components
        self.performance_benchmark = None
        self.accuracy_benchmark = None
        self.corpus_generator = None
        
        # Results storage
        self.results = {
            'performance': None,
            'accuracy': None,
            'timestamp': None,
            'config': None
        }
    
    def setup_environment(self) -> bool:
        """
        Setup test environment
        
        Returns:
            True if setup successful, False otherwise
        """
        self.logger.info("Setting up benchmark environment...")
        
        try:
            # Initialize benchmark components
            self.logger.info("Initializing benchmark components...")
            self.performance_benchmark = PerformanceBenchmark()
            self.accuracy_benchmark = AccuracyBenchmark()
            self.corpus_generator = TestCorpusGenerator()
            
            # Generate test corpus if requested
            if self.config.generate_corpus:
                self.logger.info("Generating test corpus...")
                if self.config.corpus_size == "all":
                    corpus_results = self.corpus_generator.generate_all_corpora()
                else:
                    corpus_results = self.corpus_generator.generate_corpus(self.config.corpus_size)
                
                self.logger.info(f"Test corpus generation complete: {corpus_results}")
            
            # Validate RAG pipeline is working
            self.logger.info("Validating RAG pipeline...")
            if not self.performance_benchmark.rag:
                raise RuntimeError("RAG pipeline initialization failed")
            
            # Test basic query to ensure system is working
            test_response = self.performance_benchmark.rag.query("What is machine learning?", k=1)
            if not test_response:
                raise RuntimeError("RAG pipeline test query failed")
            
            self.logger.info("Environment setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment setup failed: {e}")
            return False
    
    def _load_benchmark_queries(self) -> List[EvaluationQuery]:
        """Load benchmark queries from JSON file"""
        
        queries_file = self.test_data_dir / "benchmark_queries.json"
        
        if not queries_file.exists():
            self.logger.warning(f"Benchmark queries file not found: {queries_file}")
            # Return default queries
            return [
                EvaluationQuery(
                    query="What is machine learning?",
                    expected_topics=["algorithms", "data", "training"],
                    difficulty="easy"
                ),
                EvaluationQuery(
                    query="Compare supervised and unsupervised learning",
                    expected_topics=["labels", "classification", "clustering"],
                    difficulty="medium"
                ),
                EvaluationQuery(
                    query="How do neural networks work?",
                    expected_topics=["neurons", "weights", "layers"],
                    difficulty="medium"
                )
            ]
        
        try:
            with open(queries_file, 'r') as f:
                queries_data = json.load(f)
            
            evaluation_queries = []
            
            # Extract queries from different categories
            categories = queries_data.get('categories', {})
            
            for category_name, queries in categories.items():
                if category_name == 'edge_cases':
                    continue  # Skip edge cases for now
                
                for query_data in queries[:3]:  # Limit to first 3 from each category
                    evaluation_queries.append(EvaluationQuery(
                        query=query_data['query'],
                        expected_topics=query_data.get('expected_topics', []),
                        difficulty=query_data.get('difficulty', 'medium')
                    ))
            
            self.logger.info(f"Loaded {len(evaluation_queries)} benchmark queries")
            return evaluation_queries
            
        except Exception as e:
            self.logger.error(f"Error loading benchmark queries: {e}")
            return []
    
    def run_performance_benchmarks(self) -> Optional[List[BenchmarkResult]]:
        """Run performance benchmarks"""
        
        if not self.config.run_performance:
            return None
        
        self.logger.info("Starting performance benchmarks...")
        
        try:
            # Load test queries
            queries = [
                "What is machine learning?",
                "Explain neural networks and their applications.",
                "How does deep learning differ from traditional machine learning?",
                "What are the key components of a data science pipeline?",
                "Describe the process of training a machine learning model."
            ]
            
            # Run all performance benchmarks
            results = self.performance_benchmark.run_all_benchmarks(queries)
            
            # Save detailed results
            if self.config.save_detailed_results:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                results_file = self.reports_dir / f"performance_detailed_{timestamp}.json"
                self.performance_benchmark.save_results(str(results_file))
            
            self.logger.info("Performance benchmarks completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Performance benchmarks failed: {e}")
            return None
    
    def run_accuracy_evaluations(self) -> Optional[List[AccuracyResult]]:
        """Run accuracy evaluations"""
        
        if not self.config.run_accuracy:
            return None
        
        self.logger.info("Starting accuracy evaluations...")
        
        try:
            # Load evaluation queries
            evaluation_queries = self._load_benchmark_queries()
            
            if not evaluation_queries:
                self.logger.warning("No evaluation queries available, using defaults")
                evaluation_queries = [
                    EvaluationQuery(
                        query="What is artificial intelligence?",
                        expected_topics=["machines", "intelligence", "automation"],
                        difficulty="easy"
                    ),
                    EvaluationQuery(
                        query="How do recommendation systems work?",
                        expected_topics=["algorithms", "user preferences", "filtering"],
                        difficulty="medium"
                    )
                ]
            
            # Run all accuracy evaluations
            results = self.accuracy_benchmark.run_all_evaluations(evaluation_queries)
            
            # Save detailed results
            if self.config.save_detailed_results:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                results_file = self.reports_dir / f"accuracy_detailed_{timestamp}.json"
                self.accuracy_benchmark.save_results(str(results_file))
            
            self.logger.info("Accuracy evaluations completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Accuracy evaluations failed: {e}")
            return None
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all benchmarks
        
        Returns:
            Dictionary with all benchmark results
        """
        self.logger.info("Starting comprehensive benchmark suite...")
        
        # Setup environment
        if not self.setup_environment():
            raise RuntimeError("Environment setup failed")
        
        # Run performance benchmarks
        performance_results = self.run_performance_benchmarks()
        
        # Run accuracy evaluations
        accuracy_results = self.run_accuracy_evaluations()
        
        # Compile results
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'run_performance': self.config.run_performance,
                'run_accuracy': self.config.run_accuracy,
                'corpus_size': self.config.corpus_size,
                'num_iterations': self.config.num_performance_iterations
            },
            'performance': performance_results,
            'accuracy': accuracy_results,
            'success': True
        }
        
        self.logger.info("All benchmarks completed successfully")
        return self.results
    
    def _load_baseline_metrics(self) -> Optional[Dict[str, Any]]:
        """Load baseline metrics from file"""
        
        if not self.baseline_file.exists():
            self.logger.info("No baseline metrics file found")
            return None
        
        try:
            with open(self.baseline_file, 'r') as f:
                baseline = json.load(f)
            self.logger.info("Baseline metrics loaded successfully")
            return baseline
        except Exception as e:
            self.logger.error(f"Error loading baseline metrics: {e}")
            return None
    
    def _save_baseline_metrics(self, results: Dict[str, Any]) -> None:
        """Save current results as baseline metrics"""
        
        try:
            baseline_data = {
                'timestamp': results['timestamp'],
                'performance': {},
                'accuracy': {}
            }
            
            # Extract key metrics from performance results
            if results.get('performance'):
                for result in results['performance']:
                    if result.name == 'token_throughput' and 'tokens_per_second' in result.metrics:
                        baseline_data['performance']['tokens_per_second'] = result.metrics['tokens_per_second']
                    elif result.name == 'e2e_latency' and 'e2e_latency_ms' in result.metrics:
                        baseline_data['performance']['e2e_latency_ms'] = result.metrics['e2e_latency_ms']
                    elif result.name == 'memory_usage' and 'peak_memory_mb' in result.metrics:
                        baseline_data['performance']['memory_usage_mb'] = {'peak': result.metrics['peak_memory_mb']}
            
            # Extract key metrics from accuracy results
            if results.get('accuracy'):
                for result in results['accuracy']:
                    if result.name == 'retrieval_relevance' and 'k_5' in result.metrics:
                        baseline_data['accuracy']['precision_at_5'] = result.metrics['k_5']['precision_at_k']['mean']
                    elif result.name == 'answer_quality' and 'overall_quality' in result.metrics:
                        baseline_data['accuracy']['answer_quality'] = result.metrics['overall_quality']['mean']
            
            with open(self.baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            self.logger.info(f"Baseline metrics saved to {self.baseline_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving baseline metrics: {e}")
    
    def check_regressions(self, results: Dict[str, Any], threshold: float = 0.1) -> Dict[str, Any]:
        """
        Check for performance regressions compared to baseline
        
        Args:
            results: Current benchmark results
            threshold: Regression threshold (10% by default)
            
        Returns:
            Dictionary with regression analysis
        """
        self.logger.info("Checking for performance regressions...")
        
        baseline = self._load_baseline_metrics()
        if not baseline:
            self.logger.info("No baseline found, saving current results as baseline")
            self._save_baseline_metrics(results)
            return {'status': 'baseline_created', 'regressions': []}
        
        regressions = []
        improvements = []
        
        try:
            # Check performance regressions
            if results.get('performance') and baseline.get('performance'):
                for result in results['performance']:
                    if result.name == 'token_throughput' and 'tokens_per_second' in result.metrics:
                        current_tps = result.metrics['tokens_per_second']['mean']
                        baseline_tps = baseline['performance'].get('tokens_per_second', {}).get('mean', 0)
                        
                        if baseline_tps > 0:
                            change = (current_tps - baseline_tps) / baseline_tps
                            if change < -threshold:
                                regressions.append({
                                    'metric': 'tokens_per_second',
                                    'current': current_tps,
                                    'baseline': baseline_tps,
                                    'change_percent': change * 100
                                })
                            elif change > threshold:
                                improvements.append({
                                    'metric': 'tokens_per_second',
                                    'current': current_tps,
                                    'baseline': baseline_tps,
                                    'change_percent': change * 100
                                })
            
            # Check accuracy regressions
            if results.get('accuracy') and baseline.get('accuracy'):
                for result in results['accuracy']:
                    if result.name == 'retrieval_relevance' and 'k_5' in result.metrics:
                        current_precision = result.metrics['k_5']['precision_at_k']['mean']
                        baseline_precision = baseline['accuracy'].get('precision_at_5', 0)
                        
                        if baseline_precision > 0:
                            change = (current_precision - baseline_precision) / baseline_precision
                            if change < -threshold:
                                regressions.append({
                                    'metric': 'precision_at_5',
                                    'current': current_precision,
                                    'baseline': baseline_precision,
                                    'change_percent': change * 100
                                })
                            elif change > threshold:
                                improvements.append({
                                    'metric': 'precision_at_5',
                                    'current': current_precision,
                                    'baseline': baseline_precision,
                                    'change_percent': change * 100
                                })
            
            regression_analysis = {
                'status': 'completed',
                'threshold_percent': threshold * 100,
                'regressions': regressions,
                'improvements': improvements,
                'baseline_timestamp': baseline.get('timestamp'),
                'current_timestamp': results.get('timestamp')
            }
            
            if regressions:
                self.logger.warning(f"Found {len(regressions)} performance regressions")
            else:
                self.logger.info("No performance regressions detected")
            
            if improvements:
                self.logger.info(f"Found {len(improvements)} performance improvements")
            
            return regression_analysis
            
        except Exception as e:
            self.logger.error(f"Error checking regressions: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def generate_report(self, results: Dict[str, Any], format: str = 'markdown') -> str:
        """
        Generate benchmark report
        
        Args:
            results: Benchmark results
            format: Output format ('markdown', 'json', 'html')
            
        Returns:
            Path to generated report file
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if format == 'markdown':
            return self._generate_markdown_report(results, timestamp)
        elif format == 'json':
            return self._generate_json_report(results, timestamp)
        elif format == 'html':
            return self._generate_html_report(results, timestamp)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown_report(self, results: Dict[str, Any], timestamp: str) -> str:
        """Generate markdown benchmark report"""
        
        report_file = self.reports_dir / f"benchmark_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# RAG System Benchmark Report\n")
            f.write(f"Date: {results['timestamp']}\n\n")
            
            # Performance metrics section
            if results.get('performance'):
                f.write("## Performance Metrics\n\n")
                
                for result in results['performance']:
                    f.write(f"### {result.name.replace('_', ' ').title()}\n\n")
                    
                    if result.name == 'token_throughput' and 'tokens_per_second' in result.metrics:
                        tps = result.metrics['tokens_per_second']
                        f.write(f"- **Throughput**: {tps['mean']:.2f} ± {tps['std']:.2f} tokens/sec\n")
                        f.write(f"- **Range**: {tps['min']:.2f} - {tps['max']:.2f} tokens/sec\n")
                    
                    if result.name == 'e2e_latency' and 'e2e_latency_ms' in result.metrics:
                        latency = result.metrics['e2e_latency_ms']
                        f.write(f"- **Average Latency**: {latency['mean']:.2f}ms\n")
                        f.write(f"- **P95 Latency**: {latency.get('p95', 0):.2f}ms\n")
                    
                    if result.name == 'memory_usage' and 'peak_memory_mb' in result.metrics:
                        f.write(f"- **Peak Memory**: {result.metrics['peak_memory_mb']:.2f}MB\n")
                    
                    f.write("\n")
            
            # Accuracy metrics section
            if results.get('accuracy'):
                f.write("## Accuracy Metrics\n\n")
                
                for result in results['accuracy']:
                    f.write(f"### {result.name.replace('_', ' ').title()}\n\n")
                    
                    if result.name == 'retrieval_relevance' and 'k_5' in result.metrics:
                        precision = result.metrics['k_5']['precision_at_k']
                        f.write(f"- **Precision@5**: {precision['mean']:.3f} ± {precision['std']:.3f}\n")
                    
                    if result.name == 'answer_quality' and 'overall_quality' in result.metrics:
                        quality = result.metrics['overall_quality']
                        f.write(f"- **Answer Quality**: {quality['mean']:.3f} ± {quality['std']:.3f}\n")
                    
                    f.write("\n")
            
            # Regression analysis
            regression_analysis = self.check_regressions(results)
            f.write("## Comparison with Baseline\n\n")
            
            if regression_analysis.get('regressions'):
                f.write("### ⚠️ Regressions Detected\n\n")
                for reg in regression_analysis['regressions']:
                    f.write(f"- **{reg['metric']}**: {reg['change_percent']:.1f}% decrease\n")
                f.write("\n")
            
            if regression_analysis.get('improvements'):
                f.write("### ✅ Improvements\n\n")
                for imp in regression_analysis['improvements']:
                    f.write(f"- **{imp['metric']}**: {imp['change_percent']:.1f}% improvement\n")
                f.write("\n")
            
            if not regression_analysis.get('regressions') and not regression_analysis.get('improvements'):
                f.write("No significant changes from baseline.\n\n")
            
            # Recommendations section
            f.write("## Recommendations\n\n")
            f.write("- Monitor performance trends over time\n")
            f.write("- Investigate any significant regressions\n")
            f.write("- Consider optimizations for bottleneck areas\n")
            f.write("- Update baseline metrics after verified improvements\n")
        
        self.logger.info(f"Markdown report generated: {report_file}")
        return str(report_file)
    
    def _generate_json_report(self, results: Dict[str, Any], timestamp: str) -> str:
        """Generate JSON benchmark report"""
        
        report_file = self.reports_dir / f"benchmark_report_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if key == 'performance' and value:
                json_results[key] = []
                for result in value:
                    json_results[key].append({
                        'name': result.name,
                        'metrics': result.metrics,
                        'duration': result.duration,
                        'timestamp': result.timestamp
                    })
            elif key == 'accuracy' and value:
                json_results[key] = []
                for result in value:
                    json_results[key].append({
                        'name': result.name,
                        'metrics': result.metrics,
                        'timestamp': result.timestamp
                    })
            else:
                json_results[key] = value
        
        # Add regression analysis
        json_results['regression_analysis'] = self.check_regressions(results)
        
        with open(report_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"JSON report generated: {report_file}")
        return str(report_file)
    
    def _generate_html_report(self, results: Dict[str, Any], timestamp: str) -> str:
        """Generate HTML benchmark report"""
        
        report_file = self.reports_dir / f"benchmark_report_{timestamp}.html"
        
        # Generate basic HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG System Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .regression {{ background: #ffe6e6; }}
        .improvement {{ background: #e6ffe6; }}
    </style>
</head>
<body>
    <h1>RAG System Benchmark Report</h1>
    <p><strong>Generated:</strong> {results['timestamp']}</p>
    
    <h2>Summary</h2>
    <p>Benchmark execution completed successfully.</p>
    
    <h2>Performance Metrics</h2>
"""
        
        if results.get('performance'):
            for result in results['performance']:
                html_content += f"<div class='metric'><h3>{result.name.replace('_', ' ').title()}</h3>"
                html_content += f"<p>Duration: {result.duration:.2f}s</p></div>"
        
        html_content += "<h2>Accuracy Metrics</h2>"
        
        if results.get('accuracy'):
            for result in results['accuracy']:
                html_content += f"<div class='metric'><h3>{result.name.replace('_', ' ').title()}</h3>"
                html_content += f"<p>Evaluation completed</p></div>"
        
        html_content += """
</body>
</html>
"""
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {report_file}")
        return str(report_file)

def main():
    """Main function for running benchmarks"""
    parser = argparse.ArgumentParser(description="RAG System Benchmark Runner")
    parser.add_argument('--performance', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--accuracy', action='store_true', help='Run accuracy evaluations')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--corpus-size', choices=['small', 'medium', 'large', 'all'], 
                       default='small', help='Test corpus size')
    parser.add_argument('--format', choices=['markdown', 'json', 'html'], 
                       default='markdown', help='Report format')
    parser.add_argument('--no-corpus', action='store_true', help='Skip test corpus generation')
    parser.add_argument('--iterations', type=int, default=5, help='Number of performance iterations')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        run_performance=args.performance or args.all,
        run_accuracy=args.accuracy or args.all,
        generate_corpus=not args.no_corpus,
        corpus_size=args.corpus_size,
        num_performance_iterations=args.iterations,
        output_format=args.format
    )
    
    if not (config.run_performance or config.run_accuracy):
        print("Please specify --performance, --accuracy, or --all")
        return 1
    
    try:
        # Initialize and run benchmarks
        runner = BenchmarkRunner(config)
        results = runner.run_all_benchmarks()
        
        # Generate report
        report_file = runner.generate_report(results, args.format)
        
        print(f"\n=== Benchmark Execution Complete ===")
        print(f"Report generated: {report_file}")
        
        # Print summary
        if results.get('performance'):
            print(f"\nPerformance benchmarks: {len(results['performance'])} completed")
        
        if results.get('accuracy'):
            print(f"Accuracy evaluations: {len(results['accuracy'])} completed")
        
        return 0
        
    except Exception as e:
        print(f"Benchmark execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())