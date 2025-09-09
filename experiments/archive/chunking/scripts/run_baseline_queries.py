#!/usr/bin/env python3
"""
Baseline Query Performance Measurement Script

Runs all 30 evaluation queries to establish baseline performance metrics
for the chunking optimization experiments.
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineMeasurement:
    def __init__(self, queries_file: str, results_dir: str):
        self.queries_file = Path(queries_file)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load queries
        with open(self.queries_file, 'r') as f:
            self.queries_data = json.load(f)
        
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'queries_file': str(self.queries_file),
                'total_queries': 0,
                'configuration': {
                    'chunk_size': 512,  # Default chunk size
                    'chunk_overlap': 0,  # Default overlap
                    'retrieval_k': 5,
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'llm_model': 'gemma-3-4b-it-q4_0.gguf'
                }
            },
            'baseline_results': []
        }
    
    def run_single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single query and collect performance metrics."""
        query_id = query_data['id']
        query_text = query_data['query']
        collection = query_data['collection']
        category = query_data['category']
        expected_chunks = query_data['expected_chunks']
        
        logger.info(f"Running query {query_id}: {query_text[:50]}...")
        
        # Build command with full conda activation
        cmd = f'source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python main.py query "{query_text}" --collection {collection} --k 5 --metrics'
        
        # Time the query execution
        start_time = time.time()
        
        try:
            # Run the query with bash shell
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                executable='/bin/bash'
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if result.returncode != 0:
                logger.error(f"Query {query_id} failed with error: {result.stderr}")
                return {
                    'query_id': query_id,
                    'query_text': query_text,
                    'collection': collection,
                    'category': category,
                    'expected_chunks': expected_chunks,
                    'status': 'failed',
                    'error': result.stderr.strip(),
                    'execution_time': execution_time
                }
            
            # Parse output for metrics
            stdout = result.stdout
            stderr = result.stderr
            
            # Extract metrics from logs
            retrieval_time = self._extract_metric(stderr, "Retrieved 5 contexts in", "s")
            response_time = self._extract_metric(stderr, "Generated response", "in", "s")
            contexts_retrieved = self._extract_contexts_count(stderr)
            response_tokens = self._extract_metric(stderr, "Generated response", "(", "tokens)")
            
            # Extract answer quality indicators
            answer_start = stdout.find("Answer:")
            answer_text = stdout[answer_start:].strip() if answer_start != -1 else ""
            answer_length = len(answer_text)
            
            result_data = {
                'query_id': query_id,
                'query_text': query_text,
                'collection': collection,
                'category': category,
                'expected_chunks': expected_chunks,
                'status': 'success',
                'execution_time': execution_time,
                'metrics': {
                    'retrieval_time': retrieval_time,
                    'response_time': response_time,
                    'contexts_retrieved': contexts_retrieved,
                    'response_tokens': response_tokens,
                    'answer_length': answer_length
                },
                'answer_sample': answer_text[:200] + "..." if len(answer_text) > 200 else answer_text
            }
            
            logger.info(f"Query {query_id} completed in {execution_time:.2f}s")
            return result_data
            
        except subprocess.TimeoutExpired:
            logger.error(f"Query {query_id} timed out after 2 minutes")
            return {
                'query_id': query_id,
                'query_text': query_text,
                'collection': collection,
                'category': category,
                'expected_chunks': expected_chunks,
                'status': 'timeout',
                'execution_time': 120.0
            }
        except Exception as e:
            logger.error(f"Query {query_id} failed with exception: {e}")
            return {
                'query_id': query_id,
                'query_text': query_text,
                'collection': collection,
                'category': category,
                'expected_chunks': expected_chunks,
                'status': 'error',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _extract_metric(self, text: str, prefix: str, *suffixes) -> float:
        """Extract numeric metric from log text."""
        try:
            start_idx = text.find(prefix)
            if start_idx == -1:
                return 0.0
            
            # Find the numeric value after the prefix
            search_text = text[start_idx + len(prefix):]
            
            # Look for the number before any of the suffixes
            for suffix in suffixes:
                suffix_idx = search_text.find(suffix)
                if suffix_idx != -1:
                    # Extract number before this suffix
                    number_text = search_text[:suffix_idx].strip()
                    # Find the last number in this text
                    words = number_text.split()
                    for word in reversed(words):
                        try:
                            return float(word)
                        except ValueError:
                            continue
            
            # If no suffix found, try to extract the first number
            words = search_text.split()
            for word in words[:3]:  # Check first few words
                try:
                    return float(word.rstrip('s.,'))
                except ValueError:
                    continue
                    
        except Exception:
            pass
        
        return 0.0
    
    def _extract_contexts_count(self, text: str) -> int:
        """Extract number of retrieved contexts."""
        try:
            # Look for "Retrieved X contexts"
            import re
            match = re.search(r'Retrieved (\d+) contexts', text)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 0
    
    def run_all_queries(self) -> Dict[str, Any]:
        """Run all queries in the dataset."""
        all_queries = []
        
        # Collect all queries from both categories
        for category_name, queries in self.queries_data['queries'].items():
            all_queries.extend(queries)
        
        self.results['metadata']['total_queries'] = len(all_queries)
        
        logger.info(f"Starting baseline measurement with {len(all_queries)} queries")
        
        start_time = time.time()
        
        for i, query_data in enumerate(all_queries, 1):
            logger.info(f"Progress: {i}/{len(all_queries)}")
            
            result = self.run_single_query(query_data)
            self.results['baseline_results'].append(result)
            
            # Save intermediate results every 5 queries
            if i % 5 == 0:
                self.save_results(f"baseline_intermediate_{i}.json")
        
        total_time = time.time() - start_time
        self.results['metadata']['total_execution_time'] = total_time
        
        # Calculate summary statistics
        self._calculate_summary_stats()
        
        logger.info(f"Baseline measurement completed in {total_time:.2f}s")
        return self.results
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics from all results."""
        successful_results = [r for r in self.results['baseline_results'] if r['status'] == 'success']
        
        if not successful_results:
            logger.warning("No successful queries to calculate stats")
            return
        
        # Calculate averages
        avg_execution_time = sum(r['execution_time'] for r in successful_results) / len(successful_results)
        avg_retrieval_time = sum(r['metrics']['retrieval_time'] for r in successful_results) / len(successful_results)
        avg_response_time = sum(r['metrics']['response_time'] for r in successful_results) / len(successful_results)
        avg_contexts = sum(r['metrics']['contexts_retrieved'] for r in successful_results) / len(successful_results)
        avg_response_tokens = sum(r['metrics']['response_tokens'] for r in successful_results) / len(successful_results)
        
        # Count by status
        status_counts = {}
        for result in self.results['baseline_results']:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by collection
        collection_stats = {}
        for result in successful_results:
            collection = result['collection']
            if collection not in collection_stats:
                collection_stats[collection] = {
                    'count': 0,
                    'avg_execution_time': 0,
                    'avg_retrieval_time': 0
                }
            collection_stats[collection]['count'] += 1
        
        # Calculate per-collection averages
        for collection in collection_stats:
            collection_results = [r for r in successful_results if r['collection'] == collection]
            collection_stats[collection]['avg_execution_time'] = sum(r['execution_time'] for r in collection_results) / len(collection_results)
            collection_stats[collection]['avg_retrieval_time'] = sum(r['metrics']['retrieval_time'] for r in collection_results) / len(collection_results)
        
        self.results['summary'] = {
            'total_queries': len(self.results['baseline_results']),
            'successful_queries': len(successful_results),
            'failed_queries': len(self.results['baseline_results']) - len(successful_results),
            'success_rate': len(successful_results) / len(self.results['baseline_results']),
            'status_breakdown': status_counts,
            'performance_metrics': {
                'avg_execution_time': avg_execution_time,
                'avg_retrieval_time': avg_retrieval_time,
                'avg_response_time': avg_response_time,
                'avg_contexts_retrieved': avg_contexts,
                'avg_response_tokens': avg_response_tokens
            },
            'collection_breakdown': collection_stats
        }
    
    def save_results(self, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_results_{timestamp}.json"
        
        results_file = self.results_dir / filename
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        return results_file

def main():
    """Main execution function."""
    import os
    
    # Change to the correct working directory
    experiment_root = Path(__file__).parent.parent.parent.parent
    os.chdir(experiment_root)
    
    # Setup paths
    queries_file = "experiments/chunking/queries/chunking_queries.json"
    results_dir = "experiments/chunking/results"
    
    if not Path(queries_file).exists():
        logger.error(f"Queries file not found: {queries_file}")
        sys.exit(1)
    
    # Run baseline measurement
    baseline = BaselineMeasurement(queries_file, results_dir)
    results = baseline.run_all_queries()
    
    # Save final results
    results_file = baseline.save_results("baseline_complete.json")
    
    # Print summary
    summary = results.get('summary', {})
    logger.info("="*60)
    logger.info("BASELINE MEASUREMENT COMPLETE")
    logger.info("="*60)
    logger.info(f"Total queries: {summary.get('total_queries', 0)}")
    logger.info(f"Successful: {summary.get('successful_queries', 0)}")
    logger.info(f"Success rate: {summary.get('success_rate', 0):.1%}")
    
    if 'performance_metrics' in summary:
        metrics = summary['performance_metrics']
        logger.info(f"Average execution time: {metrics.get('avg_execution_time', 0):.2f}s")
        logger.info(f"Average retrieval time: {metrics.get('avg_retrieval_time', 0):.2f}s")
        logger.info(f"Average response time: {metrics.get('avg_response_time', 0):.2f}s")
    
    logger.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()