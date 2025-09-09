#!/usr/bin/env python3
"""
Analyze Chunking Optimization Results
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import logging

def json_serialize(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(item) for item in obj]
    else:
        return obj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_experiment_data(file_path: str) -> Dict[str, Any]:
    """Load experiment results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_configurations(results_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract and analyze configuration performance"""
    
    data = []
    
    for result in results_data.get('results', []):
        config = result.get('config', {})
        metrics = result.get('metrics', {})
        
        # Extract configuration parameters
        chunk_size = config.get('chunk_size', 512)
        chunk_overlap = config.get('chunk_overlap', 0) 
        retrieval_k = config.get('retrieval_k', 5)
        
        # Extract performance metrics
        response_time = metrics.get('response_time', 0)
        response_length = metrics.get('response_length', 0)
        num_sources = metrics.get('num_sources', 0)
        retrieval_success = metrics.get('retrieval_success', 0.0)
        response_generated = metrics.get('response_generated', 0.0)
        
        # Calculate quality score (combination of success metrics)
        quality_score = (retrieval_success + response_generated) / 2
        
        # Extract query info
        query = result.get('query', '')
        query_type = 'financial' if any(word in query.lower() 
                                      for word in ['investment', 'finance', 'money', 'market', 'fund', 'stock', 'bond', 'credit', 'tax', 'compound', 'interest']) else 'general'
        
        data.append({
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'retrieval_k': retrieval_k,
            'response_time': response_time,
            'response_length': response_length,
            'num_sources': num_sources,
            'retrieval_success': retrieval_success,
            'response_generated': response_generated, 
            'quality_score': quality_score,
            'query': query[:50] + '...' if len(query) > 50 else query,
            'query_type': query_type,
            'efficiency': quality_score / max(response_time, 0.1)  # quality per second
        })
    
    return pd.DataFrame(data)

def find_optimal_configs(df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
    """Find optimal configurations across different metrics"""
    
    # Group by configuration
    config_cols = ['chunk_size', 'chunk_overlap', 'retrieval_k']
    grouped = df.groupby(config_cols).agg({
        'response_time': ['mean', 'std'],
        'quality_score': ['mean', 'std'], 
        'efficiency': ['mean', 'std'],
        'response_length': 'mean',
        'retrieval_success': 'mean',
        'response_generated': 'mean'
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" 
                      for col in grouped.columns]
    
    results = {
        'best_overall_quality': None,
        'best_efficiency': None,
        'fastest_response': None,
        'most_reliable': None,
        'parameter_analysis': {}
    }
    
    # Best overall quality
    best_quality = grouped.loc[grouped['quality_score_mean'].idxmax()]
    results['best_overall_quality'] = {
        'config': {
            'chunk_size': int(best_quality['chunk_size']),
            'chunk_overlap': int(best_quality['chunk_overlap']),
            'retrieval_k': int(best_quality['retrieval_k'])
        },
        'quality_score': float(best_quality['quality_score_mean']),
        'response_time': float(best_quality['response_time_mean']),
        'efficiency': float(best_quality['efficiency_mean'])
    }
    
    # Best efficiency (quality per second)
    best_efficiency = grouped.loc[grouped['efficiency_mean'].idxmax()]
    results['best_efficiency'] = {
        'config': {
            'chunk_size': int(best_efficiency['chunk_size']),
            'chunk_overlap': int(best_efficiency['chunk_overlap']), 
            'retrieval_k': int(best_efficiency['retrieval_k'])
        },
        'efficiency': float(best_efficiency['efficiency_mean']),
        'quality_score': float(best_efficiency['quality_score_mean']),
        'response_time': float(best_efficiency['response_time_mean'])
    }
    
    # Fastest response
    fastest = grouped.loc[grouped['response_time_mean'].idxmin()]
    results['fastest_response'] = {
        'config': {
            'chunk_size': int(fastest['chunk_size']),
            'chunk_overlap': int(fastest['chunk_overlap']),
            'retrieval_k': int(fastest['retrieval_k'])
        },
        'response_time': float(fastest['response_time_mean']),
        'quality_score': float(fastest['quality_score_mean']),
        'efficiency': float(fastest['efficiency_mean'])
    }
    
    # Most reliable (highest success rate)
    most_reliable = grouped.loc[grouped['retrieval_success_mean'].idxmax()]
    results['most_reliable'] = {
        'config': {
            'chunk_size': int(most_reliable['chunk_size']),
            'chunk_overlap': int(most_reliable['chunk_overlap']),
            'retrieval_k': int(most_reliable['retrieval_k'])
        },
        'retrieval_success': float(most_reliable['retrieval_success_mean']),
        'response_generated': float(most_reliable['response_generated_mean']),
        'quality_score': float(most_reliable['quality_score_mean'])
    }
    
    # Parameter analysis
    for param in ['chunk_size', 'chunk_overlap', 'retrieval_k']:
        param_stats = df.groupby(param)['quality_score'].agg(['mean', 'std', 'count'])
        results['parameter_analysis'][param] = {
            'best_value': param_stats['mean'].idxmax(),
            'worst_value': param_stats['mean'].idxmin(),
            'performance_range': float(param_stats['mean'].max() - param_stats['mean'].min()),
            'stats': {str(k): {'mean': v['mean'], 'std': v['std'], 'count': v['count']} 
                     for k, v in param_stats.iterrows()}
        }
    
    return results

def generate_summary_report(df: pd.DataFrame, optimal_configs: Dict[str, Any]) -> str:
    """Generate a human-readable summary report"""
    
    report = []
    report.append("# RAG Chunking Optimization Analysis")
    report.append("=" * 50)
    report.append(f"Total experiments: {len(df)}")
    report.append(f"Unique configurations: {len(df.groupby(['chunk_size', 'chunk_overlap', 'retrieval_k']))}")
    report.append(f"Average quality score: {df['quality_score'].mean():.3f}")
    report.append(f"Average response time: {df['response_time'].mean():.2f}s")
    report.append("")
    
    report.append("## Key Findings")
    report.append("")
    
    # Best overall configuration
    best = optimal_configs['best_overall_quality']
    report.append(f"**Best Overall Configuration:**")
    report.append(f"- Chunk size: {best['config']['chunk_size']}")
    report.append(f"- Chunk overlap: {best['config']['chunk_overlap']}")
    report.append(f"- Retrieval K: {best['config']['retrieval_k']}")
    report.append(f"- Quality score: {best['quality_score']:.3f}")
    report.append(f"- Response time: {best['response_time']:.2f}s")
    report.append("")
    
    # Most efficient configuration  
    efficient = optimal_configs['best_efficiency']
    report.append(f"**Most Efficient Configuration:**")
    report.append(f"- Chunk size: {efficient['config']['chunk_size']}")
    report.append(f"- Chunk overlap: {efficient['config']['chunk_overlap']}")
    report.append(f"- Retrieval K: {efficient['config']['retrieval_k']}")
    report.append(f"- Efficiency: {efficient['efficiency']:.4f} quality/second")
    report.append(f"- Quality score: {efficient['quality_score']:.3f}")
    report.append("")
    
    # Parameter insights
    report.append("## Parameter Analysis")
    report.append("")
    
    for param, analysis in optimal_configs['parameter_analysis'].items():
        report.append(f"**{param.replace('_', ' ').title()}:**")
        report.append(f"- Best value: {analysis['best_value']}")
        report.append(f"- Performance range: {analysis['performance_range']:.3f}")
        report.append("")
    
    # Query type analysis
    if 'query_type' in df.columns:
        query_analysis = df.groupby('query_type')[['quality_score', 'response_time']].mean()
        report.append("## Query Type Performance")
        report.append("")
        for query_type, metrics in query_analysis.iterrows():
            report.append(f"**{query_type.title()} queries:**")
            report.append(f"- Quality score: {metrics['quality_score']:.3f}")
            report.append(f"- Response time: {metrics['response_time']:.2f}s")
            report.append("")
    
    return "\n".join(report)

def main():
    """Main analysis function"""
    
    results_dir = Path("experiments/chunking/results")
    
    # Process multiple experiment files
    experiment_files = {
        'FIQA': results_dir / "fiqa_sequential_optimization.json",
        'SciFact': results_dir / "scifact_sequential_optimization.json"
    }
    
    for corpus_name, file_path in experiment_files.items():
        if not file_path.exists():
            logger.warning(f"Results file not found: {file_path}")
            continue
            
        logger.info(f"Loading {corpus_name} results from {file_path}")
        corpus_data = load_experiment_data(str(file_path))
        
        # Analyze configurations
        logger.info(f"Analyzing {corpus_name} configurations...")
        df = analyze_configurations(corpus_data)
        
        if len(df) == 0:
            logger.error(f"No data found in {corpus_name} results file")
            continue
            
        logger.info(f"Loaded {len(df)} {corpus_name} experiment results")
        
        # Find optimal configurations
        logger.info(f"Finding optimal {corpus_name} configurations...")
        optimal_configs = find_optimal_configs(df)
        
        # Generate summary report
        logger.info(f"Generating {corpus_name} summary report...")
        summary = generate_summary_report(df, optimal_configs)
        
        # Save detailed results
        output_file = results_dir / f"{corpus_name.lower()}_chunking_analysis_results.json"
        with open(output_file, 'w') as f:
            data_to_save = {
                'corpus': corpus_name,
                'summary_stats': {
                    'total_experiments': len(df),
                    'avg_quality_score': df['quality_score'].mean(),
                    'avg_response_time': df['response_time'].mean(),
                    'configurations_tested': len(df.groupby(['chunk_size', 'chunk_overlap', 'retrieval_k']))
                },
                'optimal_configurations': optimal_configs,
                'parameter_distributions': {
                    'chunk_size': df['chunk_size'].value_counts().to_dict(),
                    'chunk_overlap': df['chunk_overlap'].value_counts().to_dict(), 
                    'retrieval_k': df['retrieval_k'].value_counts().to_dict()
                }
            }
            json.dump(json_serialize(data_to_save), f, indent=2)
        
        # Save summary report
        report_file = results_dir / f"{corpus_name.lower()}_chunking_optimization_report.md"
        with open(report_file, 'w') as f:
            f.write(f"# {corpus_name} RAG Chunking Optimization Analysis\n\n")
            f.write(summary)
        
        logger.info(f"{corpus_name} analysis complete!")
        logger.info(f"Detailed results: {output_file}")
        logger.info(f"Summary report: {report_file}")
        
        # Print key findings
        print(f"\n{'='*60}")
        print(f"{corpus_name.upper()} KEY FINDINGS")
        print(f"{'='*60}")
        
        best = optimal_configs['best_overall_quality']
        print(f"🏆 Best Overall: chunk_size={best['config']['chunk_size']}, "
              f"overlap={best['config']['chunk_overlap']}, k={best['config']['retrieval_k']}")
        print(f"   Quality: {best['quality_score']:.3f}, Time: {best['response_time']:.2f}s")
        
        efficient = optimal_configs['best_efficiency']  
        print(f"⚡ Most Efficient: chunk_size={efficient['config']['chunk_size']}, "
              f"overlap={efficient['config']['chunk_overlap']}, k={efficient['config']['retrieval_k']}")
        print(f"   Efficiency: {efficient['efficiency']:.4f} quality/second")
        
        print(f"\n📊 Tested {len(df)} experiments across "
              f"{len(df.groupby(['chunk_size', 'chunk_overlap', 'retrieval_k']))} configurations")
        
    # Load FIQA results for backward compatibility  
    fiqa_file = results_dir / "fiqa_sequential_optimization.json"
    if not fiqa_file.exists():
        logger.error(f"Results file not found: {fiqa_file}")
        return
    
    logger.info(f"Loading FIQA results from {fiqa_file}")
    fiqa_data = load_experiment_data(str(fiqa_file))
    
    # Analyze configurations
    logger.info("Analyzing configurations...")
    df = analyze_configurations(fiqa_data)
    
    if len(df) == 0:
        logger.error("No data found in results file")
        return
    
    logger.info(f"Loaded {len(df)} experiment results")
    
    # Find optimal configurations
    logger.info("Finding optimal configurations...")
    optimal_configs = find_optimal_configs(df)
    
    # Generate summary report
    logger.info("Generating summary report...")
    summary = generate_summary_report(df, optimal_configs)
    
    # Save detailed results
    output_file = results_dir / "chunking_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary_stats': {
                'total_experiments': len(df),
                'avg_quality_score': df['quality_score'].mean(),
                'avg_response_time': df['response_time'].mean(),
                'configurations_tested': len(df.groupby(['chunk_size', 'chunk_overlap', 'retrieval_k']))
            },
            'optimal_configurations': optimal_configs,
            'parameter_distributions': {
                'chunk_size': df['chunk_size'].value_counts().to_dict(),
                'chunk_overlap': df['chunk_overlap'].value_counts().to_dict(), 
                'retrieval_k': df['retrieval_k'].value_counts().to_dict()
            }
        }, f, indent=2)
    
    # Save summary report
    report_file = results_dir / "chunking_optimization_report.md"
    with open(report_file, 'w') as f:
        f.write(summary)
    
    logger.info(f"Analysis complete!")
    logger.info(f"Detailed results: {output_file}")
    logger.info(f"Summary report: {report_file}")
    
    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    best = optimal_configs['best_overall_quality']
    print(f"🏆 Best Overall: chunk_size={best['config']['chunk_size']}, "
          f"overlap={best['config']['chunk_overlap']}, k={best['config']['retrieval_k']}")
    print(f"   Quality: {best['quality_score']:.3f}, Time: {best['response_time']:.2f}s")
    
    efficient = optimal_configs['best_efficiency']  
    print(f"⚡ Most Efficient: chunk_size={efficient['config']['chunk_size']}, "
          f"overlap={efficient['config']['chunk_overlap']}, k={efficient['config']['retrieval_k']}")
    print(f"   Efficiency: {efficient['efficiency']:.4f} quality/second")
    
    print(f"\n📊 Tested {len(df)} experiments across "
          f"{len(df.groupby(['chunk_size', 'chunk_overlap', 'retrieval_k']))} configurations")

if __name__ == "__main__":
    main()