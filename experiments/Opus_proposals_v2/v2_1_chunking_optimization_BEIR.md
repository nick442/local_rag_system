# Research Proposal v2.1: Document Chunking Strategy Optimization with BEIR Datasets

## 1. Executive Summary

**Research Goal**: Determine optimal document chunking parameters for maximizing retrieval precision across diverse BEIR datasets (FiQA, TREC-COVID, SciFact) using the existing RAG infrastructure.

**Datasets**: 
- **FiQA**: Financial QA (57k docs, 648 queries) - tests domain-specific retrieval
- **TREC-COVID**: Scientific literature (171k docs, 50 queries) - tests scientific text chunking
- **SciFact**: Scientific claim verification (5k docs, 300 queries) - tests fact retrieval precision

**Expected Impact**: 20-30% improvement in NDCG@10 with optimized chunking per dataset type

**Timeline**: 5 days using existing infrastructure

## 2. Research Questions & Hypotheses

### Primary Research Question
What chunking parameters optimize retrieval performance across different document types in BEIR datasets?

### Hypotheses
- **H1**: Financial documents (FiQA) perform best with smaller chunks (256-512 tokens) due to precise fact density
- **H2**: Scientific articles (TREC-COVID) require larger chunks (768-1024 tokens) to maintain context
- **H3**: Claim verification (SciFact) benefits from moderate chunks (512 tokens) with high overlap (30%)

## 3. Using Existing Infrastructure

### 3.1 Dataset Preparation (Assumed Complete)

```bash
# Collections already ingested as:
# - beir_fiqa
# - beir_trec_covid  
# - beir_scifact

# Verify collections exist
python main.py collection list

# Check collection statistics
python main.py analytics stats --collection beir_fiqa
python main.py analytics stats --collection beir_trec_covid
python main.py analytics stats --collection beir_scifact
```

### 3.2 Ground Truth Preparation

Create ground truth relevance files from BEIR qrels:

```python
# scripts/prepare_beir_ground_truth.py
import json
from pathlib import Path

def convert_beir_qrels_to_ground_truth(qrels_file, output_file):
    """Convert BEIR qrels to our ground truth format."""
    ground_truth = {}
    
    with open(qrels_file) as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split('\t')
            if query_id not in ground_truth:
                ground_truth[query_id] = {}
            ground_truth[query_id][doc_id] = int(relevance)
    
    with open(output_file, 'w') as f:
        json.dump({'ground_truth': ground_truth}, f, indent=2)

# Convert each dataset's qrels
convert_beir_qrels_to_ground_truth('data/beir/fiqa/qrels.txt', 
                                   'data/ground_truth/fiqa_ground_truth.json')
convert_beir_qrels_to_ground_truth('data/beir/trec-covid/qrels.txt',
                                   'data/ground_truth/trec_covid_ground_truth.json')
convert_beir_qrels_to_ground_truth('data/beir/scifact/qrels.txt',
                                   'data/ground_truth/scifact_ground_truth.json')
```

### 3.3 Query Files Preparation

```json
// test_data/beir_fiqa_queries.json
{
  "queries": [
    {"query_id": "fiqa_1", "query": "What is the impact of quantitative easing on bond yields?"},
    {"query_id": "fiqa_2", "query": "How do I calculate the Sharpe ratio for my portfolio?"},
    // ... all FiQA queries
  ]
}

// test_data/beir_trec_covid_queries.json
{
  "queries": [
    {"query_id": "covid_1", "query": "coronavirus origin"},
    {"query_id": "covid_2", "query": "COVID-19 vaccine efficacy"},
    // ... all TREC-COVID queries
  ]
}

// test_data/beir_scifact_queries.json
{
  "queries": [
    {"query_id": "scifact_1", "query": "Smoking cessation reduces cardiovascular mortality"},
    {"query_id": "scifact_2", "query": "Vitamin D deficiency is associated with depression"},
    // ... all SciFact queries
  ]
}
```

## 4. Experimental Procedure Using Existing Infrastructure

### 4.1 Using ExperimentRunner for Parameter Sweeps

```python
# experiments/chunking/run_beir_chunking_experiment.py
from src.experiment_runner import create_experiment_runner
from src.config_manager import ExperimentConfig, ParameterRange, create_config_manager
from src.experiment_templates import create_base_experiment_config
import json

def run_chunking_experiment_for_dataset(dataset_name: str):
    """Run chunking optimization for a specific BEIR dataset."""
    
    # Load queries
    with open(f'test_data/beir_{dataset_name}_queries.json') as f:
        query_data = json.load(f)
        queries = [q['query'] for q in query_data['queries']]
    
    # Create experiment runner
    config_manager = create_config_manager()
    runner = create_experiment_runner(config_manager)
    
    # Configure base experiment
    base_config = create_base_experiment_config()
    base_config.target_corpus = f'beir_{dataset_name}'
    base_config.retrieval_method = 'vector'
    base_config.embedding_model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Define parameter ranges for chunking
    parameter_ranges = [
        ParameterRange("chunk_size", "categorical", 
                      values=[128, 256, 512, 768, 1024, 1536]),
        ParameterRange("chunk_overlap", "categorical",
                      values=[0, 32, 64, 128, 256]),
        ParameterRange("retrieval_k", "categorical",
                      values=[5, 10, 15, 20])
    ]
    
    # Run parameter sweep
    results = runner.run_parameter_sweep(
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        queries=queries[:50],  # Use subset for initial testing
        experiment_id=f'chunking_{dataset_name}_{int(time.time())}'
    )
    
    # Save results
    output_file = f'results/chunking_{dataset_name}_results.json'
    save_experiment_results(results, output_file)
    
    return results

# Run for each dataset
for dataset in ['fiqa', 'trec_covid', 'scifact']:
    print(f"Running chunking experiment for {dataset}...")
    results = run_chunking_experiment_for_dataset(dataset)
    print(f"Completed {dataset}: {len(results.results)} runs")
```

### 4.2 Using ExperimentTemplate System

```bash
# Use the existing chunk_optimization template
python main.py experiment template chunk_optimization \
  --corpus beir_fiqa \
  --queries test_data/beir_fiqa_queries.json \
  --output results/fiqa_chunking.json

python main.py experiment template chunk_optimization \
  --corpus beir_trec_covid \
  --queries test_data/beir_trec_covid_queries.json \
  --output results/trec_covid_chunking.json

python main.py experiment template chunk_optimization \
  --corpus beir_scifact \
  --queries test_data/beir_scifact_queries.json \
  --output results/scifact_chunking.json
```

### 4.3 Using CLI Parameter Sweeps

```bash
# Fine-grained chunk size sweep for FiQA
python main.py experiment sweep \
  --param chunk_size \
  --values "128,256,384,512,640,768,896,1024" \
  --queries test_data/beir_fiqa_queries.json \
  --corpus beir_fiqa \
  --output results/fiqa_chunk_size_sweep.json

# Overlap ratio sweep for TREC-COVID
python main.py experiment sweep \
  --param chunk_overlap \
  --values "0,64,128,192,256,320" \
  --queries test_data/beir_trec_covid_queries.json \
  --corpus beir_trec_covid \
  --output results/trec_covid_overlap_sweep.json

# Combined sweep for SciFact
for chunk_size in 256 512 768; do
  for overlap in 0 64 128; do
    python main.py experiment sweep \
      --param chunk_size --values $chunk_size \
      --param chunk_overlap --values $overlap \
      --queries test_data/beir_scifact_queries.json \
      --corpus beir_scifact \
      --output results/scifact_cs${chunk_size}_co${overlap}.json
  done
done
```

## 5. Result Analysis Using Existing Infrastructure

### 5.1 Using RetrievalQualityEvaluator

```python
# experiments/chunking/analyze_beir_results.py
from src.evaluation_metrics import RetrievalQualityEvaluator, ExperimentAnalyzer
from src.experiment_utils import ExperimentCollectionManager
import json
from pathlib import Path

def analyze_chunking_results(dataset_name: str):
    """Comprehensive analysis using existing evaluation infrastructure."""
    
    # Initialize evaluators
    evaluator = RetrievalQualityEvaluator()
    analyzer = ExperimentAnalyzer()
    
    # Load ground truth
    evaluator.load_ground_truth(f'data/ground_truth/{dataset_name}_ground_truth.json')
    
    # Load experiment results
    results_file = f'results/chunking_{dataset_name}_results.json'
    with open(results_file) as f:
        experiment_data = json.load(f)
    
    # Group results by configuration
    config_results = {}
    for result in experiment_data['results']:
        config = result['config']
        key = f"cs{config['chunk_size']}_co{config['chunk_overlap']}"
        
        if key not in config_results:
            config_results[key] = {
                'query_results': {},
                'response_times': [],
                'config': config
            }
        
        # Extract retrieval results
        query_id = result['query_id']
        retrieved_docs = [r['doc_id'] for r in result.get('retrieval_results', [])]
        config_results[key]['query_results'][query_id] = retrieved_docs
        config_results[key]['response_times'].append(result['duration_seconds'])
    
    # Evaluate each configuration
    evaluation_results = {}
    for config_key, data in config_results.items():
        # Calculate retrieval metrics
        metrics = evaluator.evaluate_query_set(
            data['query_results'],
            k_values=[1, 3, 5, 10, 20]
        )
        
        # Add performance metrics
        metrics['performance'] = {
            'mean_response_time': sum(data['response_times']) / len(data['response_times']),
            'config': data['config']
        }
        
        evaluation_results[config_key] = metrics
        
        # Print summary
        print(f"\n{config_key}:")
        print(f"  MRR: {metrics['mrr']:.3f}")
        print(f"  NDCG@10: {metrics['ndcg_at_k'][10]['mean']:.3f}")
        print(f"  P@5: {metrics['precision_at_k'][5]['mean']:.3f}")
        print(f"  Response Time: {metrics['performance']['mean_response_time']:.2f}s")
    
    # Find best configuration
    best_config = max(evaluation_results.items(), 
                     key=lambda x: x[1]['ndcg_at_k'][10]['mean'])
    
    print(f"\nBest configuration for {dataset_name}: {best_config[0]}")
    print(f"NDCG@10: {best_config[1]['ndcg_at_k'][10]['mean']:.3f}")
    
    # Save analysis
    output_file = f'analysis/{dataset_name}_chunking_analysis.json'
    Path('analysis').mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    return evaluation_results

# Analyze each dataset
for dataset in ['fiqa', 'trec_covid', 'scifact']:
    print(f"\nAnalyzing {dataset} results...")
    results = analyze_chunking_results(dataset)
```

### 5.2 Using StatisticalAnalyzer for Comparisons

```python
# experiments/chunking/statistical_comparison.py
from src.evaluation_metrics import StatisticalAnalyzer
import json

def compare_configurations(dataset_name: str, config_a: str, config_b: str):
    """Statistical comparison between two configurations."""
    
    analyzer = StatisticalAnalyzer()
    
    # Load analysis results
    with open(f'analysis/{dataset_name}_chunking_analysis.json') as f:
        analysis = json.load(f)
    
    # Extract NDCG scores
    scores_a = analysis[config_a]['ndcg_at_k'][10]['scores']
    scores_b = analysis[config_b]['ndcg_at_k'][10]['scores']
    
    # Perform statistical comparison
    comparison = analyzer.paired_comparison(scores_a, scores_b)
    
    print(f"\nComparing {config_a} vs {config_b} on {dataset_name}:")
    print(f"  Mean difference: {comparison['mean_difference']:.3f}")
    print(f"  Effect size: {comparison['effect_size']:.3f}")
    print(f"  T-statistic: {comparison['t_statistic']:.3f}")
    print(f"  Practical significance: {comparison['practical_significance']}")
    
    # Calculate confidence intervals
    ci_a = analyzer.confidence_interval(scores_a)
    ci_b = analyzer.confidence_interval(scores_b)
    
    print(f"  {config_a} CI: [{ci_a[0]:.3f}, {ci_a[1]:.3f}]")
    print(f"  {config_b} CI: [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")
    
    return comparison

# Compare best vs baseline for each dataset
comparisons = {
    'fiqa': ('cs512_co128', 'cs256_co64'),
    'trec_covid': ('cs1024_co256', 'cs512_co128'),
    'scifact': ('cs512_co128', 'cs768_co192')
}

for dataset, (best, baseline) in comparisons.items():
    compare_configurations(dataset, best, baseline)
```

### 5.3 Using ExperimentDatabase for Result Tracking

```python
# experiments/chunking/query_experiment_database.py
from src.experiment_runner import ExperimentDatabase
import sqlite3
import pandas as pd

def analyze_experiment_history(dataset_name: str):
    """Query experiment database for historical analysis."""
    
    db = ExperimentDatabase()
    
    # Query all chunking experiments for dataset
    query = """
    SELECT 
        e.experiment_id,
        e.created_at,
        e.total_runtime,
        er.config_json,
        AVG(CAST(json_extract(er.metrics_json, '$.response_time') AS REAL)) as avg_response_time,
        COUNT(DISTINCT er.query) as unique_queries
    FROM experiments e
    JOIN experiment_runs er ON e.experiment_id = er.experiment_id
    WHERE e.experiment_type = 'parameter_sweep'
        AND er.config_json LIKE '%"target_corpus": "beir_{}"%'
    GROUP BY e.experiment_id, json_extract(er.config_json, '$.chunk_size'), 
             json_extract(er.config_json, '$.chunk_overlap')
    ORDER BY e.created_at DESC
    """.format(dataset_name)
    
    df = pd.read_sql_query(query, sqlite3.connect('data/experiments.db'))
    
    # Analyze trends
    print(f"\nExperiment History for {dataset_name}:")
    print(f"Total experiments: {len(df)}")
    print(f"Total runtime: {df['total_runtime'].sum():.1f} seconds")
    print(f"Average queries per experiment: {df['unique_queries'].mean():.1f}")
    
    # Extract best performing configurations
    df['chunk_size'] = df['config_json'].apply(
        lambda x: json.loads(x).get('chunk_size', 0)
    )
    df['chunk_overlap'] = df['config_json'].apply(
        lambda x: json.loads(x).get('chunk_overlap', 0)
    )
    
    best_configs = df.nsmallest(5, 'avg_response_time')[
        ['chunk_size', 'chunk_overlap', 'avg_response_time']
    ]
    
    print("\nFastest configurations:")
    print(best_configs.to_string())
    
    return df
```

## 6. Visualization Using Existing Tools

```python
# experiments/chunking/visualize_results.py
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

def create_chunking_heatmap(dataset_name: str):
    """Create heatmap of NDCG scores for chunk size/overlap combinations."""
    
    # Load analysis results
    with open(f'analysis/{dataset_name}_chunking_analysis.json') as f:
        analysis = json.load(f)
    
    # Extract data for heatmap
    chunk_sizes = sorted(set(
        res['config']['chunk_size'] 
        for res in analysis.values()
    ))
    overlaps = sorted(set(
        res['config']['chunk_overlap']
        for res in analysis.values()
    ))
    
    # Create matrix
    ndcg_matrix = np.zeros((len(chunk_sizes), len(overlaps)))
    
    for config_key, results in analysis.items():
        cs_idx = chunk_sizes.index(results['config']['chunk_size'])
        co_idx = overlaps.index(results['config']['chunk_overlap'])
        ndcg_matrix[cs_idx, co_idx] = results['ndcg_at_k'][10]['mean']
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(ndcg_matrix, 
                xticklabels=overlaps,
                yticklabels=chunk_sizes,
                annot=True, 
                fmt='.3f',
                cmap='viridis')
    plt.xlabel('Chunk Overlap (tokens)')
    plt.ylabel('Chunk Size (tokens)')
    plt.title(f'{dataset_name.upper()} - NDCG@10 Heatmap')
    plt.tight_layout()
    plt.savefig(f'figures/{dataset_name}_chunking_heatmap.png', dpi=300)
    plt.show()

# Generate heatmaps for all datasets
for dataset in ['fiqa', 'trec_covid', 'scifact']:
    create_chunking_heatmap(dataset)
```

## 7. Step-by-Step Implementation Guide

### Day 1: Setup and Baseline
```bash
# 1. Verify BEIR datasets are ingested
python main.py collection list | grep beir

# 2. Create ground truth files
python scripts/prepare_beir_ground_truth.py

# 3. Run baseline experiments
python main.py query "test query" --collection beir_fiqa --metrics
python main.py query "test query" --collection beir_trec_covid --metrics
python main.py query "test query" --collection beir_scifact --metrics
```

### Day 2-3: Run Experiments
```bash
# 1. Run chunking template for each dataset
for dataset in fiqa trec_covid scifact; do
  python main.py experiment template chunk_optimization \
    --corpus beir_$dataset \
    --queries test_data/beir_${dataset}_queries.json \
    --output results/${dataset}_chunking.json
done

# 2. Monitor progress
tail -f logs/rag_system.log | grep experiment

# 3. Check experiment database
sqlite3 data/experiments.db "SELECT experiment_id, status, completed_runs, total_runs FROM experiments ORDER BY created_at DESC LIMIT 5"
```

### Day 4: Analysis
```bash
# 1. Run evaluation scripts
python experiments/chunking/analyze_beir_results.py

# 2. Statistical comparisons
python experiments/chunking/statistical_comparison.py

# 3. Query experiment history
python experiments/chunking/query_experiment_database.py
```

### Day 5: Visualization and Report
```bash
# 1. Generate visualizations
python experiments/chunking/visualize_results.py

# 2. Export final results
python main.py experiment list --status completed --limit 50 > reports/experiment_list.txt

# 3. Generate report
python experiments/chunking/generate_report.py
```

## 8. Expected Outcomes

### Dataset-Specific Optimal Configurations

| Dataset | Optimal Chunk Size | Optimal Overlap | Expected NDCG@10 |
|---------|-------------------|-----------------|------------------|
| FiQA | 256-384 tokens | 64 tokens (20%) | 0.35-0.40 |
| TREC-COVID | 768-1024 tokens | 192 tokens (25%) | 0.65-0.70 |
| SciFact | 512 tokens | 128 tokens (25%) | 0.60-0.65 |

### Performance Metrics
- **Retrieval Latency**: <200ms for optimal configurations
- **Memory Usage**: <2GB per collection with embeddings
- **Indexing Speed**: 100-150 docs/second

## 9. Validation Using Existing Tools

```python
# experiments/chunking/validate_optimal_config.py
from src.experiment_utils import ExperimentCollectionManager

def validate_optimal_configurations():
    """Validate that optimal configurations meet requirements."""
    
    manager = ExperimentCollectionManager('data/rag_vectors.db')
    
    optimal_configs = {
        'fiqa': 'exp_cs256_co64',
        'trec_covid': 'exp_cs1024_co256',
        'scifact': 'exp_cs512_co128'
    }
    
    for dataset, collection_id in optimal_configs.items():
        validation = manager.validate_collection_parameters(collection_id)
        
        print(f"\n{dataset} - {collection_id}:")
        print(f"  Valid: {validation['valid']}")
        print(f"  Chunks: {validation['total_chunks']}")
        print(f"  Documents: {validation['total_documents']}")
        print(f"  Warnings: {validation.get('warnings', [])}")
        
        # Check collection size
        size_mb = manager.get_collection_size(collection_id)
        print(f"  Size: {size_mb:.2f} MB")
```

## 10. Success Criteria

- ✅ Complete parameter sweep for all 3 BEIR datasets
- ✅ Achieve statistically significant improvements (p < 0.05)
- ✅ Document optimal configurations per dataset type
- ✅ Generate reproducible results with <5% variance
- ✅ Create reusable analysis pipeline for future experiments

---

**Status**: Ready for implementation with existing infrastructure
**Prerequisites**: BEIR datasets ingested as collections
**Next Step**: Run Day 1 baseline experiments
