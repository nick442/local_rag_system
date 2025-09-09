#!/usr/bin/env python3
"""
Find Optimal Chunking Configurations

Identifies the top-performing chunking configurations from experiment results
based on specified metrics and thresholds.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimalConfigFinder:
    """Finds optimal chunking configurations from experiment results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = self._load_all_results()
        
    def _load_all_results(self) -> List[Dict[str, Any]]:
        """Load all experiment results from JSON files in directory."""
        all_results = []
        
        for results_file in self.results_dir.glob("*.json"):
            if results_file.name.startswith("baseline"):
                continue  # Skip baseline files for this analysis
                
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    if 'results' in data:
                        for result in data['results']:
                            result['source_file'] = str(results_file)
                            all_results.append(result)
                    logger.info(f"Loaded {len(data.get('results', []))} results from {results_file}")
            except Exception as e:
                logger.warning(f"Could not load {results_file}: {e}")
        
        logger.info(f"Total results loaded: {len(all_results)}")
        return all_results
    
    def extract_performance_data(self) -> pd.DataFrame:
        """Extract performance data into a structured DataFrame."""
        data = []
        
        for result in self.results:
            if result.get('status') != 'success':
                continue
                
            config = result.get('configuration', {})
            metrics = result.get('metrics', {})
            
            # Extract key configuration parameters
            chunk_size = config.get('chunk_size', 512)
            chunk_overlap = config.get('chunk_overlap', 0)
            chunking_strategy = config.get('chunking_strategy', 'token')
            
            # Extract performance metrics
            retrieval_metrics = metrics.get('retrieval_metrics', {})
            response_quality = metrics.get('response_quality', {})
            
            # Calculate composite performance score
            precision_5 = retrieval_metrics.get('precision@5', 0.0)
            recall_5 = retrieval_metrics.get('recall@5', 0.0)
            ndcg_10 = retrieval_metrics.get('ndcg@10', 0.0)
            response_score = response_quality.get('score', 0.0)
            
            # Weighted composite score
            composite_score = (0.4 * precision_5 + 0.3 * recall_5 + 
                             0.2 * ndcg_10 + 0.1 * response_score)
            
            data.append({
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'chunking_strategy': chunking_strategy,
                'query_id': result.get('query_id', 'unknown'),
                'query_category': result.get('query_category', 'unknown'),
                'corpus': result.get('corpus', 'unknown'),
                'precision@5': precision_5,
                'recall@5': recall_5,  
                'ndcg@10': ndcg_10,
                'mrr': retrieval_metrics.get('mrr', 0.0),
                'response_quality': response_score,
                'composite_score': composite_score,
                'execution_time': result.get('execution_time', 0.0),
                'source_file': result.get('source_file', 'unknown')
            })
        
        return pd.DataFrame(data)
    
    def find_top_configurations(self, df: pd.DataFrame, 
                              metric: str = "composite_score",
                              threshold: float = 0.8,
                              n_top: int = 10) -> List[Dict[str, Any]]:
        """Find top N configurations that exceed the threshold."""
        
        # Group by configuration
        config_cols = ['chunk_size', 'chunk_overlap', 'chunking_strategy']
        grouped = df.groupby(config_cols).agg({
            metric: ['mean', 'std', 'count'],
            'execution_time': 'mean',
            'precision@5': 'mean',
            'recall@5': 'mean',
            'ndcg@10': 'mean',
            'response_quality': 'mean'
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" 
                          for col in grouped.columns]
        
        # Filter by threshold
        threshold_mask = grouped[f'{metric}_mean'] >= threshold
        top_configs = grouped[threshold_mask].copy()
        
        if len(top_configs) == 0:
            logger.warning(f"No configurations found above threshold {threshold}")
            # Lower threshold and try again
            threshold = grouped[f'{metric}_mean'].quantile(0.8)
            logger.info(f"Using 80th percentile threshold: {threshold:.3f}")
            threshold_mask = grouped[f'{metric}_mean'] >= threshold
            top_configs = grouped[threshold_mask].copy()
        
        # Sort by performance
        top_configs = top_configs.sort_values(f'{metric}_mean', ascending=False)
        
        # Convert to list of dictionaries
        results = []
        for _, row in top_configs.head(n_top).iterrows():
            config = {
                'configuration': {
                    'chunk_size': int(row['chunk_size']),
                    'chunk_overlap': int(row['chunk_overlap']),
                    'chunking_strategy': row['chunking_strategy']
                },
                'performance': {
                    f'{metric}_mean': float(row[f'{metric}_mean']),
                    f'{metric}_std': float(row[f'{metric}_std']),
                    'precision@5_mean': float(row['precision@5_mean']),
                    'recall@5_mean': float(row['recall@5_mean']),
                    'ndcg@10_mean': float(row['ndcg@10_mean']),
                    'response_quality_mean': float(row['response_quality_mean']),
                    'execution_time_mean': float(row['execution_time_mean'])
                },
                'sample_size': int(row[f'{metric}_count']),
                'meets_threshold': float(row[f'{metric}_mean']) >= threshold
            }
            results.append(config)
        
        return results
    
    def analyze_parameter_importance(self, df: pd.DataFrame, 
                                   metric: str = "composite_score") -> Dict[str, Any]:
        """Analyze the importance of different parameters."""
        
        analysis = {}
        
        # Chunk size analysis
        chunk_size_stats = df.groupby('chunk_size')[metric].agg(['mean', 'std', 'count'])
        analysis['chunk_size'] = {
            'statistics': {int(k): {'mean': v['mean'], 'std': v['std'], 'count': v['count']} 
                          for k, v in chunk_size_stats.iterrows()},
            'best_value': int(chunk_size_stats['mean'].idxmax()),
            'worst_value': int(chunk_size_stats['mean'].idxmin()),
            'range': float(chunk_size_stats['mean'].max() - chunk_size_stats['mean'].min())
        }
        
        # Chunk overlap analysis
        overlap_stats = df.groupby('chunk_overlap')[metric].agg(['mean', 'std', 'count'])
        analysis['chunk_overlap'] = {
            'statistics': {int(k): {'mean': v['mean'], 'std': v['std'], 'count': v['count']} 
                          for k, v in overlap_stats.iterrows()},
            'best_value': int(overlap_stats['mean'].idxmax()),
            'worst_value': int(overlap_stats['mean'].idxmin()),
            'range': float(overlap_stats['mean'].max() - overlap_stats['mean'].min())
        }
        
        # Chunking strategy analysis
        strategy_stats = df.groupby('chunking_strategy')[metric].agg(['mean', 'std', 'count'])
        analysis['chunking_strategy'] = {
            'statistics': {k: {'mean': v['mean'], 'std': v['std'], 'count': v['count']} 
                          for k, v in strategy_stats.iterrows()},
            'best_value': strategy_stats['mean'].idxmax(),
            'worst_value': strategy_stats['mean'].idxmin(),
            'range': float(strategy_stats['mean'].max() - strategy_stats['mean'].min())
        }
        
        return analysis
    
    def find_pareto_optimal(self, df: pd.DataFrame, 
                          performance_metric: str = "composite_score",
                          cost_metric: str = "execution_time") -> List[Dict[str, Any]]:
        """Find Pareto-optimal configurations (best performance vs. execution time trade-off)."""
        
        # Group by configuration
        config_cols = ['chunk_size', 'chunk_overlap', 'chunking_strategy']
        grouped = df.groupby(config_cols).agg({
            performance_metric: 'mean',
            cost_metric: 'mean'
        }).reset_index()
        
        # Find Pareto frontier (maximize performance, minimize cost)
        pareto_optimal = []
        
        for i, row_i in grouped.iterrows():
            is_pareto = True
            perf_i = row_i[performance_metric]
            cost_i = row_i[cost_metric]
            
            for j, row_j in grouped.iterrows():
                if i == j:
                    continue
                    
                perf_j = row_j[performance_metric]
                cost_j = row_j[cost_metric]
                
                # Check if j dominates i (better or equal performance, lower or equal cost)
                if perf_j >= perf_i and cost_j <= cost_i and (perf_j > perf_i or cost_j < cost_i):
                    is_pareto = False
                    break
            
            if is_pareto:
                config = {
                    'configuration': {
                        'chunk_size': int(row_i['chunk_size']),
                        'chunk_overlap': int(row_i['chunk_overlap']),
                        'chunking_strategy': row_i['chunking_strategy']
                    },
                    'performance': float(row_i[performance_metric]),
                    'cost': float(row_i[cost_metric]),
                    'efficiency_ratio': float(row_i[performance_metric] / max(row_i[cost_metric], 0.01))
                }
                pareto_optimal.append(config)
        
        # Sort by efficiency ratio
        pareto_optimal.sort(key=lambda x: x['efficiency_ratio'], reverse=True)
        
        return pareto_optimal
    
    def generate_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate practical recommendations based on analysis."""
        
        recommendations = {
            'best_overall': None,
            'best_for_speed': None,
            'best_for_accuracy': None,
            'balanced_choice': None,
            'parameter_guidelines': {}
        }
        
        # Best overall (composite score)
        top_configs = self.find_top_configurations(df, metric="composite_score", n_top=1)
        if top_configs:
            recommendations['best_overall'] = top_configs[0]
        
        # Best for speed (lowest execution time with decent performance)
        speed_df = df[df['composite_score'] >= df['composite_score'].quantile(0.7)]
        if len(speed_df) > 0:
            speed_config = speed_df.loc[speed_df['execution_time'].idxmin()]
            recommendations['best_for_speed'] = {
                'configuration': {
                    'chunk_size': int(speed_config['chunk_size']),
                    'chunk_overlap': int(speed_config['chunk_overlap']),
                    'chunking_strategy': speed_config['chunking_strategy']
                },
                'execution_time': float(speed_config['execution_time']),
                'composite_score': float(speed_config['composite_score'])
            }
        
        # Best for accuracy (highest precision@5)
        accuracy_config = df.loc[df['precision@5'].idxmax()]
        recommendations['best_for_accuracy'] = {
            'configuration': {
                'chunk_size': int(accuracy_config['chunk_size']),
                'chunk_overlap': int(accuracy_config['chunk_overlap']),
                'chunking_strategy': accuracy_config['chunking_strategy']
            },
            'precision@5': float(accuracy_config['precision@5']),
            'execution_time': float(accuracy_config['execution_time'])
        }
        
        # Balanced choice (Pareto optimal with good efficiency)
        pareto_configs = self.find_pareto_optimal(df)
        if pareto_configs:
            recommendations['balanced_choice'] = pareto_configs[0]  # Highest efficiency ratio
        
        # Parameter guidelines
        param_importance = self.analyze_parameter_importance(df)
        recommendations['parameter_guidelines'] = param_importance
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Find optimal chunking configurations')
    parser.add_argument('--results', required=True, help='Directory containing experiment results')
    parser.add_argument('--metric', default='composite_score', help='Primary metric for optimization')
    parser.add_argument('--threshold', type=float, default=0.8, help='Performance threshold')
    parser.add_argument('--n-top', type=int, default=10, help='Number of top configurations to return')
    parser.add_argument('--output', required=True, help='Output file for optimal configurations')
    
    args = parser.parse_args()
    
    finder = OptimalConfigFinder(args.results)
    df = finder.extract_performance_data()
    
    if len(df) == 0:
        logger.error("No performance data found in results directory")
        return
    
    logger.info(f"Analyzing {len(df)} experiment results")
    
    # Find top configurations
    top_configs = finder.find_top_configurations(
        df, metric=args.metric, threshold=args.threshold, n_top=args.n_top
    )
    
    # Find Pareto-optimal configurations
    pareto_configs = finder.find_pareto_optimal(df)
    
    # Generate recommendations
    recommendations = finder.generate_recommendations(df)
    
    # Prepare output
    output = {
        'metadata': {
            'total_experiments': len(df),
            'unique_configurations': len(df.groupby(['chunk_size', 'chunk_overlap', 'chunking_strategy'])),
            'primary_metric': args.metric,
            'threshold': args.threshold,
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'top_configurations': top_configs,
        'pareto_optimal_configurations': pareto_configs,
        'recommendations': recommendations,
        'parameter_analysis': finder.analyze_parameter_importance(df, args.metric)
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {args.output}")
    logger.info(f"Found {len(top_configs)} configurations above threshold")
    logger.info(f"Found {len(pareto_configs)} Pareto-optimal configurations")

if __name__ == "__main__":
    main()