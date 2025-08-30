"""
Evaluation Metrics for RAG System Performance Assessment
Comprehensive metrics framework for measuring retrieval quality and system performance.
"""

import math
import statistics
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path


class RetrievalQualityEvaluator:
    """Comprehensive retrieval quality evaluation using standard IR metrics."""
    
    def __init__(self, ground_truth_relevance: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize with ground truth relevance judgments.
        
        Args:
            ground_truth_relevance: Format: {query_id: {doc_id: relevance_score}}
                                  Relevance scores: 0=not relevant, 1=partially relevant, 2=highly relevant
        """
        self.ground_truth = ground_truth_relevance or {}
    
    def load_ground_truth(self, file_path: str) -> None:
        """Load ground truth relevance judgments from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'ground_truth' in data:
                self.ground_truth = data['ground_truth']
            else:
                self.ground_truth = data
    
    def calculate_precision_at_k(self, query_id: str, retrieved_docs: List[str], k: int) -> float:
        """
        Calculate Precision@K for a single query.
        
        Args:
            query_id: Query identifier
            retrieved_docs: List of retrieved document IDs in ranking order
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0.0-1.0)
        """
        if not retrieved_docs or k <= 0:
            return 0.0
            
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k 
                               if self.ground_truth.get(query_id, {}).get(doc, 0) > 0)
        return relevant_in_top_k / k
    
    def calculate_recall_at_k(self, query_id: str, retrieved_docs: List[str], k: int) -> float:
        """
        Calculate Recall@K for a single query.
        
        Args:
            query_id: Query identifier
            retrieved_docs: List of retrieved document IDs in ranking order
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0.0-1.0)
        """
        relevant_docs = [doc for doc, score in self.ground_truth.get(query_id, {}).items() 
                        if score > 0]
        if not relevant_docs:
            return 0.0
            
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_in_top_k / len(relevant_docs)
    
    def calculate_mrr(self, query_results: Dict[str, List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank across all queries.
        
        Args:
            query_results: {query_id: [retrieved_doc_ids_in_order]}
            
        Returns:
            MRR score (0.0-1.0)
        """
        reciprocal_ranks = []
        
        for query_id, retrieved_docs in query_results.items():
            for rank, doc in enumerate(retrieved_docs, 1):
                if self.ground_truth.get(query_id, {}).get(doc, 0) > 0:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
                
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_ndcg_at_k(self, query_id: str, retrieved_docs: List[str], k: int) -> float:
        """
        Calculate NDCG@K for a single query.
        
        Args:
            query_id: Query identifier
            retrieved_docs: List of retrieved document IDs in ranking order
            k: Number of top results to consider
            
        Returns:
            NDCG@K score (0.0-1.0)
        """
        def dcg_at_k(relevance_scores: List[float], k: int) -> float:
            """Calculate Discounted Cumulative Gain at K."""
            return sum(score / math.log2(i + 2) for i, score in enumerate(relevance_scores[:k]))
        
        # Get relevance scores for retrieved documents
        retrieved_relevance = [self.ground_truth.get(query_id, {}).get(doc, 0) 
                              for doc in retrieved_docs[:k]]
        
        # Calculate DCG@K
        dcg = dcg_at_k(retrieved_relevance, k)
        
        # Calculate IDCG@K (ideal DCG)
        ideal_relevance = sorted(self.ground_truth.get(query_id, {}).values(), reverse=True)
        idcg = dcg_at_k(ideal_relevance, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_query_set(self, query_results: Dict[str, List[str]], 
                          k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        Evaluate a complete set of queries with comprehensive metrics.
        
        Args:
            query_results: {query_id: [retrieved_doc_ids_in_order]}
            k_values: List of K values for P@K, R@K, NDCG@K calculations
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            'total_queries': len(query_results),
            'mrr': self.calculate_mrr(query_results),
            'precision_at_k': {},
            'recall_at_k': {},
            'ndcg_at_k': {},
            'per_query_results': {}
        }
        
        # Calculate metrics for each K value
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for query_id, retrieved_docs in query_results.items():
                p_at_k = self.calculate_precision_at_k(query_id, retrieved_docs, k)
                r_at_k = self.calculate_recall_at_k(query_id, retrieved_docs, k)
                ndcg_at_k = self.calculate_ndcg_at_k(query_id, retrieved_docs, k)
                
                precision_scores.append(p_at_k)
                recall_scores.append(r_at_k)
                ndcg_scores.append(ndcg_at_k)
                
                # Store per-query results
                if query_id not in results['per_query_results']:
                    results['per_query_results'][query_id] = {}
                results['per_query_results'][query_id][f'precision_at_{k}'] = p_at_k
                results['per_query_results'][query_id][f'recall_at_{k}'] = r_at_k
                results['per_query_results'][query_id][f'ndcg_at_{k}'] = ndcg_at_k
            
            # Store aggregate metrics
            results['precision_at_k'][k] = {
                'mean': statistics.mean(precision_scores) if precision_scores else 0.0,
                'std': statistics.stdev(precision_scores) if len(precision_scores) > 1 else 0.0,
                'scores': precision_scores
            }
            results['recall_at_k'][k] = {
                'mean': statistics.mean(recall_scores) if recall_scores else 0.0,
                'std': statistics.stdev(recall_scores) if len(recall_scores) > 1 else 0.0,
                'scores': recall_scores
            }
            results['ndcg_at_k'][k] = {
                'mean': statistics.mean(ndcg_scores) if ndcg_scores else 0.0,
                'std': statistics.stdev(ndcg_scores) if len(ndcg_scores) > 1 else 0.0,
                'scores': ndcg_scores
            }
        
        return results


class StatisticalAnalyzer:
    """Statistical analysis for chunking experiments."""
    
    @staticmethod
    def paired_comparison(config_a_results: List[float], 
                         config_b_results: List[float]) -> Dict[str, float]:
        """
        Paired t-test comparison between configurations.
        
        Args:
            config_a_results: Metric values for configuration A
            config_b_results: Metric values for configuration B
            
        Returns:
            Statistical comparison results
        """
        if len(config_a_results) != len(config_b_results):
            raise ValueError("Results lists must have the same length for paired comparison")
        
        # Calculate differences
        differences = [a - b for a, b in zip(config_a_results, config_b_results)]
        
        if len(differences) < 2:
            return {
                'mean_difference': differences[0] if differences else 0.0,
                'effect_size': 0.0,
                'sample_size': len(differences),
                'insufficient_data': True
            }
        
        mean_diff = statistics.mean(differences)
        std_diff = statistics.stdev(differences)
        
        # Calculate Cohen's d for paired samples
        effect_size = mean_diff / std_diff if std_diff > 0 else 0.0
        
        # Simple significance test (t-statistic approximation)
        n = len(differences)
        t_stat = mean_diff / (std_diff / math.sqrt(n)) if std_diff > 0 else 0.0
        
        return {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'effect_size': effect_size,
            't_statistic': t_stat,
            'sample_size': n,
            'practical_significance': abs(effect_size) > 0.3,
            'large_effect': abs(effect_size) > 0.8
        }
    
    @staticmethod
    def confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for metric.
        
        Args:
            data: List of metric values
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) < 2:
            mean_val = data[0] if data else 0.0
            return (mean_val, mean_val)
        
        mean_val = statistics.mean(data)
        std_err = statistics.stdev(data) / math.sqrt(len(data))
        
        # Use t-distribution critical value approximation
        # For 95% CI and reasonable sample sizes, use ~1.96
        t_critical = 2.0 if len(data) < 30 else 1.96
        margin = t_critical * std_err
        
        return (mean_val - margin, mean_val + margin)
    
    @staticmethod
    def summary_statistics(data: List[float]) -> Dict[str, float]:
        """
        Calculate summary statistics for a dataset.
        
        Args:
            data: List of numeric values
            
        Returns:
            Dictionary of summary statistics
        """
        if not data:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        return {
            'count': len(data),
            'mean': statistics.mean(data),
            'std': statistics.stdev(data) if len(data) > 1 else 0.0,
            'min': min(data),
            'max': max(data),
            'median': statistics.median(data),
            'cv': statistics.stdev(data) / statistics.mean(data) if len(data) > 1 and statistics.mean(data) != 0 else 0.0
        }


class ExperimentAnalyzer:
    """High-level analyzer for chunking experiments."""
    
    def __init__(self):
        self.retrieval_evaluator = RetrievalQualityEvaluator()
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def analyze_experiment_results(self, results_file: str, 
                                 ground_truth_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze experiment results with comprehensive metrics.
        
        Args:
            results_file: Path to experiment results JSON
            ground_truth_file: Optional path to ground truth relevance judgments
            
        Returns:
            Comprehensive analysis results
        """
        # Load experiment results
        with open(results_file, 'r') as f:
            experiment_data = json.load(f)
        
        # Load ground truth if provided
        if ground_truth_file:
            self.retrieval_evaluator.load_ground_truth(ground_truth_file)
        
        # Group results by configuration
        config_results = {}
        for result in experiment_data.get('results', []):
            config_key = self._get_config_key(result.get('config', {}))
            if config_key not in config_results:
                config_results[config_key] = []
            config_results[config_key].append(result)
        
        # Analyze each configuration
        analysis = {
            'total_configurations': len(config_results),
            'configurations': {},
            'best_configurations': {},
            'statistical_comparisons': {}
        }
        
        for config_key, results in config_results.items():
            config_analysis = self._analyze_configuration(config_key, results)
            analysis['configurations'][config_key] = config_analysis
        
        # Find best configurations for each metric
        analysis['best_configurations'] = self._identify_best_configurations(analysis['configurations'])
        
        return analysis
    
    def _get_config_key(self, config: Dict[str, Any]) -> str:
        """Generate a unique key for a configuration."""
        chunk_size = config.get('chunk_size', 'default')
        chunk_overlap = config.get('chunk_overlap', 'default')
        return f"cs{chunk_size}_co{chunk_overlap}"
    
    def _analyze_configuration(self, config_key: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results for a single configuration."""
        # Extract metrics
        response_times = [r.get('duration_seconds', 0) for r in results]
        prompt_tokens = [r.get('prompt_tokens', 0) for r in results if r.get('prompt_tokens')]
        output_tokens = [r.get('output_tokens', 0) for r in results if r.get('output_tokens')]
        
        analysis = {
            'config_key': config_key,
            'total_runs': len(results),
            'performance_metrics': {
                'response_time': self.statistical_analyzer.summary_statistics(response_times),
                'prompt_tokens': self.statistical_analyzer.summary_statistics(prompt_tokens),
                'output_tokens': self.statistical_analyzer.summary_statistics(output_tokens)
            }
        }
        
        return analysis
    
    def _identify_best_configurations(self, configurations: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Identify best performing configurations for each metric."""
        best_configs = {}
        
        # Find configuration with lowest mean response time
        min_response_time = float('inf')
        best_response_config = None
        
        for config_key, analysis in configurations.items():
            response_time_mean = analysis['performance_metrics']['response_time']['mean']
            if response_time_mean < min_response_time:
                min_response_time = response_time_mean
                best_response_config = config_key
        
        best_configs['fastest_response'] = best_response_config
        
        return best_configs