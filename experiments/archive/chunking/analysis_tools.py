#!/usr/bin/env python3
"""
Analysis and visualization tools for chunking optimization experiments.

This module provides statistical analysis, visualization, and reporting
capabilities for the chunking strategy optimization experiments.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.stats import ttest_rel, friedmanchisquare
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StatisticalResult:
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    significant: bool
    practical_significance: bool
    interpretation: str

@dataclass
class OptimalConfiguration:
    """Optimal configuration result."""
    chunk_size: int
    chunk_overlap: int
    collection: str
    metric_value: float
    metric_name: str
    confidence_interval: Tuple[float, float]
    rank: int

class ChunkingAnalyzer:
    """Comprehensive analysis tool for chunking experiments."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Analysis configuration
        self.significance_level = 0.05
        self.min_effect_size = 0.5  # Cohen's d threshold
        
    def load_experiment_data(self, session_id: Optional[str] = None) -> pd.DataFrame:
        """Load experiment metrics data from JSONL files."""
        if session_id:
            metrics_files = list(self.results_dir.glob(f"*metrics*{session_id}*.jsonl"))
        else:
            # Load most recent metrics file
            metrics_files = list(self.results_dir.glob("*metrics*.jsonl"))
            if not metrics_files:
                raise FileNotFoundError("No metrics files found")
            metrics_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not metrics_files:
            raise FileNotFoundError(f"No metrics files found for session {session_id}")
        
        # Load and combine all metrics
        all_data = []
        for file in metrics_files[:3]:  # Use up to 3 most recent files
            with open(file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Filter for experiment metrics (not ingestion metrics)
                        if 'run_id' in data and 'query' in data:
                            all_data.append(data)
                    except json.JSONDecodeError:
                        continue
        
        if not all_data:
            raise ValueError("No valid experiment data found")
        
        df = pd.DataFrame(all_data)
        
        # Data cleaning and type conversion
        numeric_columns = [
            'chunk_size', 'chunk_overlap', 'response_time_seconds', 
            'retrieval_time_seconds', 'generation_time_seconds', 
            'retrieved_chunks', 'response_length_chars', 'response_length_words',
            'tokens_per_second', 'memory_usage_mb'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter out failed experiments for main analysis
        df = df[df['success'] == True].copy()
        
        print(f"Loaded {len(df)} successful experiments from {len(metrics_files)} files")
        return df

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        summary = {
            'total_experiments': len(df),
            'unique_configurations': len(df.groupby(['chunk_size', 'chunk_overlap'])),
            'collections_tested': df['collection'].nunique(),
            'chunk_sizes_tested': sorted(df['chunk_size'].unique().tolist()),
            'overlap_values_tested': sorted(df['chunk_overlap'].unique().tolist()),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            }
        }
        
        # Performance metrics summary
        perf_metrics = ['response_time_seconds', 'tokens_per_second', 'memory_usage_mb']
        summary['performance_summary'] = {}
        
        for metric in perf_metrics:
            if metric in df.columns:
                summary['performance_summary'][metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'median': float(df[metric].median())
                }
        
        # Configuration performance ranking
        config_perf = df.groupby(['chunk_size', 'chunk_overlap']).agg({
            'response_time_seconds': ['mean', 'std', 'count'],
            'tokens_per_second': 'mean',
            'memory_usage_mb': 'mean'
        }).round(3)
        
        # Flatten column names
        config_perf.columns = ['_'.join(col).strip() for col in config_perf.columns]
        config_perf = config_perf.reset_index()
        
        # Rank by response time (lower is better)
        config_perf['rank'] = config_perf['response_time_seconds_mean'].rank()
        summary['top_configurations'] = config_perf.nsmallest(5, 'response_time_seconds_mean').to_dict('records')
        
        return summary

    def find_optimal_configurations(self, df: pd.DataFrame, 
                                   metric: str = 'response_time_seconds',
                                   minimize: bool = True) -> List[OptimalConfiguration]:
        """Find optimal configurations based on specified metric."""
        
        # Group by configuration
        grouped = df.groupby(['chunk_size', 'chunk_overlap', 'collection'])
        
        results = []
        for (chunk_size, overlap, collection), group in grouped:
            if len(group) < 3:  # Need minimum samples for confidence interval
                continue
            
            values = group[metric].values
            mean_val = np.mean(values)
            
            # Calculate confidence interval
            if len(values) > 1:
                ci = stats.t.interval(
                    0.95, len(values)-1, 
                    loc=mean_val, 
                    scale=stats.sem(values)
                )
            else:
                ci = (mean_val, mean_val)
            
            config = OptimalConfiguration(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                collection=collection,
                metric_value=mean_val,
                metric_name=metric,
                confidence_interval=ci,
                rank=0  # Will be set later
            )
            results.append(config)
        
        # Rank configurations
        results.sort(key=lambda x: x.metric_value, reverse=not minimize)
        for i, config in enumerate(results, 1):
            config.rank = i
        
        return results

    def perform_statistical_tests(self, df: pd.DataFrame) -> List[StatisticalResult]:
        """Perform statistical significance tests between configurations."""
        results = []
        
        # Test effect of chunk size on response time
        chunk_sizes = sorted(df['chunk_size'].unique())
        if len(chunk_sizes) >= 3:
            groups = [df[df['chunk_size'] == size]['response_time_seconds'].values 
                     for size in chunk_sizes]
            
            # Remove empty groups
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 3:
                try:
                    stat, p_val = friedmanchisquare(*groups)
                    
                    # Calculate effect size (eta-squared approximation)
                    n_total = sum(len(g) for g in groups)
                    k = len(groups)
                    effect_size = (stat - k + 1) / (n_total - k)
                    
                    result = StatisticalResult(
                        test_name="Chunk Size Effect (Friedman)",
                        statistic=stat,
                        p_value=p_val,
                        effect_size=effect_size,
                        significant=p_val < self.significance_level,
                        practical_significance=effect_size > 0.1,
                        interpretation=f"{'Significant' if p_val < self.significance_level else 'Non-significant'} effect of chunk size on response time"
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error in Friedman test: {e}")
        
        # Pairwise comparisons for top configurations
        top_configs = df.nsmallest(100, 'response_time_seconds')  # Top 100 fastest
        
        if len(top_configs) >= 20:  # Need sufficient data
            config_groups = top_configs.groupby(['chunk_size', 'chunk_overlap'])
            group_data = [(name, group['response_time_seconds'].values) 
                         for name, group in config_groups if len(group) >= 3]
            
            if len(group_data) >= 2:
                # Compare best two configurations
                (name1, data1), (name2, data2) = group_data[:2]
                
                try:
                    stat, p_val = ttest_rel(data1[:min(len(data1), len(data2))], 
                                           data2[:min(len(data1), len(data2))])
                    
                    # Cohen's d for paired samples
                    effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt(
                        (np.var(data1) + np.var(data2)) / 2
                    )
                    
                    result = StatisticalResult(
                        test_name=f"Top Configs Comparison: {name1} vs {name2}",
                        statistic=stat,
                        p_value=p_val,
                        effect_size=abs(effect_size),
                        significant=p_val < self.significance_level,
                        practical_significance=abs(effect_size) > self.min_effect_size,
                        interpretation=f"{'Significant' if p_val < self.significance_level else 'Non-significant'} difference between top configurations"
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error in t-test: {e}")
        
        return results

    def create_visualizations(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create comprehensive visualizations of experiment results."""
        viz_files = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Response Time Heatmap
        if len(df) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create pivot table for heatmap
            heatmap_data = df.groupby(['chunk_size', 'chunk_overlap'])['response_time_seconds'].mean().unstack(fill_value=np.nan)
            
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax)
            ax.set_title('Average Response Time by Chunk Size and Overlap', fontsize=16, fontweight='bold')
            ax.set_xlabel('Chunk Overlap (tokens)', fontsize=12)
            ax.set_ylabel('Chunk Size (tokens)', fontsize=12)
            
            heatmap_file = self.figures_dir / 'response_time_heatmap.png'
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files['heatmap'] = str(heatmap_file)
        
        # 2. Performance Distribution Plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['response_time_seconds', 'tokens_per_second', 'memory_usage_mb', 'response_length_words']
        titles = ['Response Time (seconds)', 'Tokens per Second', 'Memory Usage (MB)', 'Response Length (words)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in df.columns:
                sns.boxplot(data=df, x='chunk_size', y=metric, ax=axes[i])
                axes[i].set_title(title, fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        dist_file = self.figures_dir / 'performance_distributions.png'
        plt.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files['distributions'] = str(dist_file)
        
        # 3. Optimization Curves
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Response time vs chunk size
        chunk_perf = df.groupby('chunk_size').agg({
            'response_time_seconds': ['mean', 'std'],
            'tokens_per_second': 'mean'
        })
        chunk_perf.columns = ['_'.join(col).strip() for col in chunk_perf.columns]
        
        axes[0].errorbar(chunk_perf.index, chunk_perf['response_time_seconds_mean'], 
                        yerr=chunk_perf['response_time_seconds_std'], 
                        marker='o', capsize=5, capthick=2)
        axes[0].set_xlabel('Chunk Size (tokens)')
        axes[0].set_ylabel('Response Time (seconds)')
        axes[0].set_title('Response Time vs Chunk Size')
        axes[0].grid(True, alpha=0.3)
        
        # Tokens per second vs chunk size
        axes[1].plot(chunk_perf.index, chunk_perf['tokens_per_second_mean'], 
                    marker='s', linewidth=2, markersize=8)
        axes[1].set_xlabel('Chunk Size (tokens)')
        axes[1].set_ylabel('Tokens per Second')
        axes[1].set_title('Generation Speed vs Chunk Size')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_file = self.figures_dir / 'optimization_curves.png'
        plt.savefig(curves_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files['curves'] = str(curves_file)
        
        # 4. Memory Usage Analysis
        if 'memory_usage_mb' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Scatter plot of memory vs chunk size, colored by response time
            scatter = ax.scatter(df['chunk_size'], df['memory_usage_mb'], 
                               c=df['response_time_seconds'], cmap='viridis', 
                               alpha=0.6, s=50)
            
            ax.set_xlabel('Chunk Size (tokens)')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage vs Chunk Size (colored by response time)')
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('Response Time (seconds)')
            
            memory_file = self.figures_dir / 'memory_analysis.png'
            plt.savefig(memory_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files['memory'] = str(memory_file)
        
        return viz_files

    def generate_comprehensive_report(self, df: pd.DataFrame, 
                                    output_file: Optional[str] = None) -> str:
        """Generate comprehensive markdown report of analysis results."""
        
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.results_dir / f'chunking_analysis_report_{timestamp}.md'
        
        # Perform all analyses
        summary = self.generate_summary_statistics(df)
        optimal_configs = self.find_optimal_configurations(df)
        statistical_results = self.perform_statistical_tests(df)
        visualizations = self.create_visualizations(df)
        
        # Generate report content
        report_lines = [
            "# Document Chunking Strategy Optimization - Analysis Report\n",
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Analysis Period:** {summary['date_range']['start']} to {summary['date_range']['end']}\n",
            
            "## Executive Summary\n",
            f"- **Total Experiments:** {summary['total_experiments']:,}",
            f"- **Unique Configurations:** {summary['unique_configurations']:,}",
            f"- **Collections Tested:** {summary['collections_tested']}",
            f"- **Chunk Sizes:** {summary['chunk_sizes_tested']}",
            f"- **Overlap Values:** {summary['overlap_values_tested']}\n",
            
            "## Performance Summary\n"
        ]
        
        for metric, stats in summary['performance_summary'].items():
            report_lines.extend([
                f"### {metric.replace('_', ' ').title()}",
                f"- **Mean:** {stats['mean']:.3f}",
                f"- **Std Dev:** {stats['std']:.3f}",
                f"- **Range:** {stats['min']:.3f} - {stats['max']:.3f}",
                f"- **Median:** {stats['median']:.3f}\n"
            ])
        
        # Top configurations
        report_lines.extend([
            "## Top Performing Configurations\n",
            "| Rank | Chunk Size | Overlap | Avg Response Time | Tokens/sec | Memory (MB) | Experiments |",
            "|------|------------|---------|-------------------|------------|-------------|-------------|"
        ])
        
        for i, config in enumerate(summary['top_configurations'], 1):
            report_lines.append(
                f"| {i} | {int(config['chunk_size'])} | {int(config['chunk_overlap'])} | "
                f"{config['response_time_seconds_mean']:.3f}s | "
                f"{config.get('tokens_per_second_mean', 0):.1f} | "
                f"{config.get('memory_usage_mb_mean', 0):.1f} | "
                f"{int(config['response_time_seconds_count'])} |"
            )
        
        # Statistical analysis
        report_lines.extend(["\n## Statistical Analysis\n"])
        
        for result in statistical_results:
            report_lines.extend([
                f"### {result.test_name}",
                f"- **Test Statistic:** {result.statistic:.3f}",
                f"- **P-value:** {result.p_value:.6f}",
                f"- **Effect Size:** {result.effect_size:.3f}",
                f"- **Significant:** {'Yes' if result.significant else 'No'}",
                f"- **Practical Significance:** {'Yes' if result.practical_significance else 'No'}",
                f"- **Interpretation:** {result.interpretation}\n"
            ])
        
        # Optimal configurations details
        report_lines.extend(["\n## Detailed Optimal Configurations\n"])
        
        for config in optimal_configs[:10]:  # Top 10
            report_lines.extend([
                f"### Rank {config.rank}: {config.chunk_size} tokens, {config.chunk_overlap} overlap",
                f"- **Collection:** {config.collection}",
                f"- **{config.metric_name}:** {config.metric_value:.3f}",
                f"- **95% CI:** ({config.confidence_interval[0]:.3f}, {config.confidence_interval[1]:.3f})\n"
            ])
        
        # Visualizations
        if visualizations:
            report_lines.extend(["\n## Visualizations\n"])
            for viz_type, viz_path in visualizations.items():
                report_lines.append(f"- **{viz_type.title()}:** `{viz_path}`")
        
        # Recommendations
        report_lines.extend([
            "\n## Recommendations\n",
            "Based on the analysis of chunking strategy optimization:\n"
        ])
        
        if optimal_configs:
            best_config = optimal_configs[0]
            report_lines.extend([
                f"1. **Optimal Configuration:** Use {best_config.chunk_size} token chunks with {best_config.chunk_overlap} token overlap",
                f"2. **Expected Performance:** ~{best_config.metric_value:.3f}s average response time",
                "3. **Implementation:** Update configuration files with optimal parameters",
                "4. **Monitoring:** Continue monitoring performance with production workloads\n"
            ])
        
        report_lines.extend([
            "## Methodology\n",
            "- **Experimental Design:** Parameter sweep across chunk sizes and overlap ratios",
            "- **Metrics Collected:** Response time, tokens/second, memory usage, response quality",
            "- **Statistical Tests:** Friedman test for multiple groups, paired t-tests for comparisons",
            "- **Significance Level:** α = 0.05",
            "- **Effect Size Threshold:** Cohen's d > 0.5 for practical significance\n",
            
            "---\n",
            "*This report was generated automatically by the chunking optimization analysis system.*"
        ])
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Comprehensive analysis report saved to: {output_file}")
        return str(output_file)

def main():
    """Example usage of the analysis tools."""
    analyzer = ChunkingAnalyzer("experiments/chunking/logs")
    
    try:
        # Load data
        df = analyzer.load_experiment_data()
        print(f"Loaded {len(df)} experiment records")
        
        # Generate comprehensive report
        report_file = analyzer.generate_comprehensive_report(df)
        print(f"Analysis complete! Report saved to: {report_file}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()