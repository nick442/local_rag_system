#!/usr/bin/env python3
"""
Statistical Validation for Chunking Optimization Experiments

Implements statistical tests and effect size calculations to validate
the significance of performance differences between chunking configurations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import bootstrap
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalValidator:
    """Validates statistical significance of chunking experiment results."""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results = self._load_results()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load experiment results from JSON file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def extract_performance_metrics(self, metric_name: str = "precision@5") -> pd.DataFrame:
        """Extract performance metrics from results."""
        data = []
        
        for result in self.results.get('results', []):
            if result.get('status') == 'success' and 'metrics' in result:
                config = result['configuration']
                metrics = result['metrics']
                
                # Extract retrieval metrics if available
                retrieval_metrics = metrics.get('retrieval_metrics', {})
                performance_value = retrieval_metrics.get(metric_name, 0.0)
                
                data.append({
                    'chunk_size': config.get('chunk_size', 512),
                    'chunk_overlap': config.get('chunk_overlap', 0),
                    'chunking_strategy': config.get('chunking_strategy', 'token'),
                    'query_id': result.get('query_id', 'unknown'),
                    'query_category': result.get('query_category', 'unknown'),
                    'performance': performance_value,
                    'execution_time': result.get('execution_time', 0),
                    'response_quality': metrics.get('response_quality', {}).get('score', 0)
                })
        
        return pd.DataFrame(data)
    
    def bootstrap_ci(self, data: np.ndarray, n_bootstrap: int = 1000, 
                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence intervals."""
        def mean_statistic(x):
            return np.mean(x)
        
        # Reshape data for bootstrap
        data_reshaped = data.reshape(-1, 1)
        
        # Perform bootstrap
        res = bootstrap((data_reshaped.ravel(),), mean_statistic, 
                       n_resamples=n_bootstrap, confidence_level=confidence,
                       random_state=42)
        
        return res.confidence_interval.low, res.confidence_interval.high
    
    def validate_significance(self, results_a: np.ndarray, results_b: np.ndarray,
                            alpha: float = 0.05) -> Dict[str, Any]:
        """
        Ensure statistical significance of comparisons between two configurations.
        
        Args:
            results_a: Performance values for configuration A
            results_b: Performance values for configuration B  
            alpha: Significance level (default 0.05)
            
        Returns:
            Dictionary with statistical test results
        """
        # Remove NaN values
        results_a = results_a[~np.isnan(results_a)]
        results_b = results_b[~np.isnan(results_b)]
        
        if len(results_a) == 0 or len(results_b) == 0:
            return {"error": "Insufficient data for comparison"}
        
        # Paired t-test (if same length) or independent t-test
        if len(results_a) == len(results_b):
            t_stat, p_value = stats.ttest_rel(results_a, results_b)
            test_type = "paired_ttest"
        else:
            t_stat, p_value = stats.ttest_ind(results_a, results_b)
            test_type = "independent_ttest"
        
        # Effect size (Cohen's d)
        if len(results_a) == len(results_b):
            pooled_std = np.std(results_a - results_b)
            cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std if pooled_std > 0 else 0
        else:
            pooled_std = np.sqrt(((len(results_a) - 1) * np.var(results_a, ddof=1) + 
                                (len(results_b) - 1) * np.var(results_b, ddof=1)) / 
                               (len(results_a) + len(results_b) - 2))
            cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std if pooled_std > 0 else 0
        
        # Bootstrap confidence intervals for the difference
        if len(results_a) == len(results_b):
            diff = results_a - results_b
            ci_lower, ci_upper = self.bootstrap_ci(diff)
        else:
            # For independent samples, bootstrap the difference in means
            n_bootstrap = 1000
            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                sample_a = np.random.choice(results_a, len(results_a), replace=True)
                sample_b = np.random.choice(results_b, len(results_b), replace=True)
                bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))
            
            bootstrap_diffs = np.array(bootstrap_diffs)
            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Effect size interpretation
        effect_size_interpretation = "negligible"
        if abs(cohens_d) >= 0.8:
            effect_size_interpretation = "large"
        elif abs(cohens_d) >= 0.5:
            effect_size_interpretation = "medium"
        elif abs(cohens_d) >= 0.2:
            effect_size_interpretation = "small"
        
        return {
            "test_type": test_type,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "effect_size": float(cohens_d),
            "effect_size_interpretation": effect_size_interpretation,
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "practical_significance": abs(cohens_d) > 0.5,
            "mean_a": float(np.mean(results_a)),
            "mean_b": float(np.mean(results_b)),
            "std_a": float(np.std(results_a)),
            "std_b": float(np.std(results_b)),
            "n_a": len(results_a),
            "n_b": len(results_b)
        }
    
    def compare_configurations(self, df: pd.DataFrame, 
                             config_a: Dict[str, Any], config_b: Dict[str, Any],
                             metric: str = "performance") -> Dict[str, Any]:
        """Compare two specific configurations."""
        
        # Filter data for configuration A
        mask_a = True
        for key, value in config_a.items():
            if key in df.columns:
                mask_a &= (df[key] == value)
        data_a = df[mask_a][metric].values
        
        # Filter data for configuration B  
        mask_b = True
        for key, value in config_b.items():
            if key in df.columns:
                mask_b &= (df[key] == value)
        data_b = df[mask_b][metric].values
        
        if len(data_a) == 0:
            return {"error": f"No data found for configuration A: {config_a}"}
        if len(data_b) == 0:
            return {"error": f"No data found for configuration B: {config_b}"}
        
        result = self.validate_significance(data_a, data_b)
        result["config_a"] = config_a
        result["config_b"] = config_b
        
        return result
    
    def find_best_configurations(self, df: pd.DataFrame, 
                               metric: str = "performance", 
                               n_top: int = 5) -> List[Dict[str, Any]]:
        """Find the top N configurations by performance."""
        
        # Group by configuration and calculate mean performance
        config_cols = ['chunk_size', 'chunk_overlap', 'chunking_strategy']
        grouped = df.groupby(config_cols)[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Sort by mean performance
        grouped = grouped.sort_values('mean', ascending=False)
        
        top_configs = []
        for _, row in grouped.head(n_top).iterrows():
            config = {col: row[col] for col in config_cols}
            top_configs.append({
                'configuration': config,
                'mean_performance': float(row['mean']),
                'std_performance': float(row['std']),
                'n_samples': int(row['count'])
            })
        
        return top_configs
    
    def analyze_parameter_effects(self, df: pd.DataFrame, 
                                metric: str = "performance") -> Dict[str, Any]:
        """Analyze the effect of individual parameters."""
        
        analysis = {}
        
        # Analyze chunk_size effect
        chunk_size_stats = df.groupby('chunk_size')[metric].agg(['mean', 'std', 'count'])
        analysis['chunk_size'] = {
            'stats': chunk_size_stats.to_dict('index'),
            'best_value': int(chunk_size_stats['mean'].idxmax()),
            'worst_value': int(chunk_size_stats['mean'].idxmin())
        }
        
        # Analyze chunk_overlap effect  
        overlap_stats = df.groupby('chunk_overlap')[metric].agg(['mean', 'std', 'count'])
        analysis['chunk_overlap'] = {
            'stats': overlap_stats.to_dict('index'),
            'best_value': int(overlap_stats['mean'].idxmax()),
            'worst_value': int(overlap_stats['mean'].idxmin())
        }
        
        # Analyze chunking_strategy effect
        strategy_stats = df.groupby('chunking_strategy')[metric].agg(['mean', 'std', 'count'])
        analysis['chunking_strategy'] = {
            'stats': strategy_stats.to_dict('index'),
            'best_value': strategy_stats['mean'].idxmax(),
            'worst_value': strategy_stats['mean'].idxmin()
        }
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='Statistical validation of chunking experiments')
    parser.add_argument('--results', required=True, help='Path to experiment results JSON file')
    parser.add_argument('--metric', default='precision@5', help='Metric to analyze')
    parser.add_argument('--output', help='Output file for analysis results')
    parser.add_argument('--compare', nargs=2, metavar=('CONFIG_A', 'CONFIG_B'), 
                       help='Compare two configurations (JSON format)')
    
    args = parser.parse_args()
    
    validator = StatisticalValidator(args.results)
    df = validator.extract_performance_metrics(args.metric)
    
    logger.info(f"Loaded {len(df)} data points for analysis")
    
    analysis_results = {
        'metadata': {
            'results_file': args.results,
            'metric': args.metric,
            'total_data_points': len(df)
        },
        'parameter_effects': validator.analyze_parameter_effects(df),
        'top_configurations': validator.find_best_configurations(df),
    }
    
    # Perform specific comparison if requested
    if args.compare:
        try:
            config_a = json.loads(args.compare[0])
            config_b = json.loads(args.compare[1])
            comparison = validator.compare_configurations(df, config_a, config_b)
            analysis_results['configuration_comparison'] = comparison
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration JSON: {e}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        logger.info(f"Analysis results saved to {args.output}")
    else:
        print(json.dumps(analysis_results, indent=2))

if __name__ == "__main__":
    main()