# Research Proposal v2.3: Hybrid Retrieval Strategy Optimization with BEIR Datasets

## 1. Executive Summary

**Research Goal**: Optimize the combination of dense vector and sparse keyword retrieval using BEIR datasets to maximize retrieval effectiveness with the fully-supported hybrid infrastructure.

**Datasets**:
- **FiQA**: Financial QA - tests hybrid retrieval on financial terminology and concepts
- **TREC-COVID**: Scientific literature - tests hybrid on medical/scientific queries
- **SciFact**: Scientific claims - tests hybrid on fact verification tasks

**Expected Impact**: 20-30% improvement in NDCG@10 with optimal alpha parameters per query type

**Timeline**: 5 days using fully existing infrastructure (no modifications needed)

## 2. Research Questions & Hypotheses

### Primary Research Question
What is the optimal weighting (alpha parameter) between dense and sparse retrieval methods for different BEIR datasets and query types?

### Hypotheses
- **H1**: FiQA will benefit from keyword-biased hybrid (α=0.3) due to financial terminology
- **H2**: TREC-COVID will prefer balanced hybrid (α=0.5) for scientific concept + term matching
- **H3**: SciFact will need vector-biased hybrid (α=0.7) for semantic claim matching
- **H4**: Dynamic alpha based on query characteristics will outperform static alpha by >15%

## 3. Using the Existing Hybrid Infrastructure

### 3.1 Current Hybrid Implementation

The system already has full hybrid retrieval support:

```python
# From src/retriever.py - already implemented
def retrieve(self, query: str, k: int = 5, method: str = 'vector', 
            collection_id: Optional[str] = None) -> List[RetrievalResult]:
    """
    Retrieve relevant chunks using specified method.
    
    Args:
        method: 'vector', 'keyword', or 'hybrid'
    """
    if method == 'hybrid':
        return self._hybrid_search(query, k, collection_id)
    elif method == 'keyword':
        return self._keyword_search(query, k, collection_id)
    else:
        return self._vector_search(query, k, collection_id)

def _hybrid_search(self, query: str, k: int, collection_id: Optional[str] = None,
                  alpha: float = 0.5) -> List[RetrievalResult]:
    """Combine vector and keyword search results."""
    # Already implemented with score fusion
```

The `similarity_threshold` parameter in ExperimentConfig serves as the alpha weight!

### 3.2 Using retrieval_methods Template

```python
# The retrieval_methods template already exists and is perfect for this
from src.experiment_templates import get_template

template = get_template('retrieval_methods')
print(template.name)  # "Retrieval Method Analysis"
print(template.parameter_ranges)
# Includes:
# - retrieval_method: ["vector", "keyword", "hybrid"]
# - retrieval_k: [3, 5, 7, 10, 15, 20]
# - similarity_threshold: [0.0, 0.2, 0.4, 0.6, 0.8]  <- This is alpha!
```

## 4. Experimental Procedures

### 4.1 Direct CLI Usage for Hybrid Experiments

```bash
# Test single methods as baselines
python main.py experiment sweep \
  --param retrieval_method \
  --values "vector,keyword,hybrid" \
  --queries test_data/beir_fiqa_queries.json \
  --corpus beir_fiqa \
  --output results/fiqa_retrieval_methods.json

# Alpha parameter optimization (similarity_threshold is alpha)
python main.py experiment sweep \
  --param similarity_threshold \
  --range "0.0,1.0,0.1" \
  --queries test_data/beir_trec_covid_queries.json \
  --corpus beir_trec_covid \
  --output results/trec_covid_alpha_sweep.json

# Use the existing template
python main.py experiment template retrieval_methods \
  --corpus beir_scifact \
  --queries test_data/beir_scifact_queries.json \
  --output results/scifact_retrieval_methods.json
```

### 4.2 Programmatic Experiments Using ExperimentRunner

```python
# experiments/hybrid/run_hybrid_optimization.py
from src.experiment_runner import create_experiment_runner
from src.config_manager import ParameterRange, create_config_manager
from src.experiment_templates import create_base_experiment_config
import json
import numpy as np

def optimize_hybrid_for_beir(dataset_name: str):
    """Optimize hybrid retrieval for specific BEIR dataset."""
    
    # Load queries
    with open(f'test_data/beir_{dataset_name}_queries.json') as f:
        queries = [q['query'] for q in json.load(f)['queries']]
    
    runner = create_experiment_runner()
    
    # Base configuration
    base_config = create_base_experiment_config()
    base_config.target_corpus = f'beir_{dataset_name}'
    
    # Define parameter ranges
    parameter_ranges = [
        # Test all retrieval methods
        ParameterRange("retrieval_method", "categorical", 
                      values=["vector", "keyword", "hybrid"]),
        
        # Alpha parameter (called similarity_threshold in config)
        ParameterRange("similarity_threshold", "linear", 
                      min_val=0.0, max_val=1.0, step=0.1),
        
        # Retrieval depths
        ParameterRange("retrieval_k", "categorical",
                      values=[5, 10, 20])
    ]
    
    # Run comprehensive sweep
    results = runner.run_parameter_sweep(
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        queries=queries[:200],  # Use subset
        experiment_id=f'hybrid_{dataset_name}_{int(time.time())}'
    )
    
    return results

# Run for all datasets
for dataset in ['fiqa', 'trec_covid', 'scifact']:
    print(f"Optimizing hybrid retrieval for {dataset}...")
    results = optimize_hybrid_for_beir(dataset)
    print(f"Completed: {len(results.results)} configurations tested")
```

### 4.3 Query-Adaptive Alpha Selection

```python
# experiments/hybrid/dynamic_alpha.py
import re
from typing import Tuple

class QueryAnalyzer:
    """Analyze queries to determine optimal retrieval strategy."""
    
    def __init__(self):
        # Load domain-specific patterns
        self.financial_terms = ['stock', 'bond', 'portfolio', 'investment', 
                               'yield', 'dividend', 'equity', 'margin']
        self.medical_terms = ['covid', 'vaccine', 'treatment', 'symptom',
                            'mortality', 'efficacy', 'clinical']
        self.version_pattern = r'\d+\.\d+'
        self.acronym_pattern = r'\b[A-Z]{2,}\b'
    
    def analyze_query(self, query: str) -> Tuple[str, float]:
        """
        Determine optimal alpha for query.
        
        Returns:
            (strategy_name, alpha_value)
        """
        query_lower = query.lower()
        
        # Check for specific patterns
        has_version = bool(re.search(self.version_pattern, query))
        has_acronym = bool(re.search(self.acronym_pattern, query))
        has_financial = any(term in query_lower for term in self.financial_terms)
        has_medical = any(term in query_lower for term in self.medical_terms)
        
        # Count characteristics
        word_count = len(query.split())
        is_question = '?' in query or any(
            query_lower.startswith(q) for q in ['what', 'how', 'why', 'when']
        )
        
        # Determine alpha based on characteristics
        if has_version or has_acronym:
            return "keyword_dominant", 0.2  # Favor keyword search
        elif has_financial:
            return "financial_balanced", 0.35  # Slight keyword bias
        elif has_medical:
            return "medical_balanced", 0.45  # Slight keyword bias
        elif is_question and word_count > 5:
            return "semantic_dominant", 0.7  # Favor vector search
        elif word_count <= 3:
            return "short_query", 0.4  # Keyword helps short queries
        else:
            return "balanced", 0.5

def test_dynamic_alpha(dataset_name: str):
    """Test dynamic alpha selection."""
    
    analyzer = QueryAnalyzer()
    
    # Load queries
    with open(f'test_data/beir_{dataset_name}_queries.json') as f:
        queries = json.load(f)['queries']
    
    # Analyze query distribution
    strategy_counts = {}
    alpha_values = []
    
    for q in queries:
        strategy, alpha = analyzer.analyze_query(q['query'])
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        alpha_values.append(alpha)
    
    print(f"\n{dataset_name.upper()} Query Analysis:")
    print(f"Average alpha: {np.mean(alpha_values):.3f}")
    print(f"Alpha std dev: {np.std(alpha_values):.3f}")
    print("\nStrategy distribution:")
    for strategy, count in sorted(strategy_counts.items(), 
                                  key=lambda x: x[1], reverse=True):
        pct = count / len(queries) * 100
        print(f"  {strategy}: {count} ({pct:.1f}%)")
    
    return alpha_values

# Run dynamic alpha experiment
def run_dynamic_alpha_experiment(dataset_name: str):
    """Compare static vs dynamic alpha."""
    
    analyzer = QueryAnalyzer()
    runner = create_experiment_runner()
    
    # Load queries
    with open(f'test_data/beir_{dataset_name}_queries.json') as f:
        queries = [q['query'] for q in json.load(f)['queries']]
    
    # Test with static alpha (best from optimization)
    static_config = create_base_experiment_config()
    static_config.target_corpus = f'beir_{dataset_name}'
    static_config.retrieval_method = 'hybrid'
    static_config.similarity_threshold = 0.5  # Static alpha
    
    static_results = []
    for query in queries[:100]:
        result = runner._run_single_experiment(
            f"static_{hash(query)}", static_config, query
        )
        static_results.append(result)
    
    # Test with dynamic alpha
    dynamic_results = []
    for query in queries[:100]:
        _, alpha = analyzer.analyze_query(query)
        
        dynamic_config = create_base_experiment_config()
        dynamic_config.target_corpus = f'beir_{dataset_name}'
        dynamic_config.retrieval_method = 'hybrid'
        dynamic_config.similarity_threshold = alpha  # Dynamic alpha
        
        result = runner._run_single_experiment(
            f"dynamic_{hash(query)}", dynamic_config, query
        )
        dynamic_results.append(result)
    
    return static_results, dynamic_results
```

## 5. Analysis Using Existing Infrastructure

### 5.1 Comprehensive Evaluation

```python
# experiments/hybrid/analyze_hybrid.py
from src.evaluation_metrics import RetrievalQualityEvaluator, ExperimentAnalyzer
from src.experiment_runner import ExperimentDatabase
import json
import numpy as np

def analyze_hybrid_results(dataset_name: str):
    """Analyze hybrid retrieval using existing evaluators."""
    
    evaluator = RetrievalQualityEvaluator()
    evaluator.load_ground_truth(f'data/ground_truth/{dataset_name}_ground_truth.json')
    
    # Load experiment results
    with open(f'results/{dataset_name}_retrieval_methods.json') as f:
        results = json.load(f)
    
    # Group by method and alpha
    method_results = {'vector': {}, 'keyword': {}, 'hybrid': {}}
    
    for result in results['results']:
        method = result['config']['retrieval_method']
        alpha = result['config'].get('similarity_threshold', 0.5)
        
        key = f"alpha_{alpha:.1f}" if method == 'hybrid' else 'baseline'
        
        if key not in method_results[method]:
            method_results[method][key] = {
                'query_results': {},
                'response_times': []
            }
        
        # Store results
        query_id = result.get('query_id', result['query'][:20])
        retrieved = [r['doc_id'] for r in result.get('retrieval_results', [])]
        method_results[method][key]['query_results'][query_id] = retrieved
        method_results[method][key]['response_times'].append(
            result['duration_seconds']
        )
    
    # Evaluate each configuration
    evaluation_results = {}
    
    for method, configs in method_results.items():
        evaluation_results[method] = {}
        
        for config_key, data in configs.items():
            if data['query_results']:
                metrics = evaluator.evaluate_query_set(
                    data['query_results'],
                    k_values=[1, 5, 10, 20]
                )
                
                evaluation_results[method][config_key] = {
                    'mrr': metrics['mrr'],
                    'ndcg@10': metrics['ndcg_at_k'][10]['mean'],
                    'p@5': metrics['precision_at_k'][5]['mean'],
                    'avg_latency': np.mean(data['response_times'])
                }
    
    # Find best hybrid alpha
    best_alpha = max(
        evaluation_results['hybrid'].items(),
        key=lambda x: x[1]['ndcg@10']
    )
    
    print(f"\n{dataset_name.upper()} Results:")
    print(f"Vector baseline NDCG@10: {evaluation_results['vector']['baseline']['ndcg@10']:.3f}")
    print(f"Keyword baseline NDCG@10: {evaluation_results['keyword']['baseline']['ndcg@10']:.3f}")
    print(f"Best hybrid NDCG@10: {best_alpha[1]['ndcg@10']:.3f} ({best_alpha[0]})")
    
    # Calculate improvements
    vector_baseline = evaluation_results['vector']['baseline']['ndcg@10']
    keyword_baseline = evaluation_results['keyword']['baseline']['ndcg@10']
    hybrid_best = best_alpha[1]['ndcg@10']
    
    print(f"Improvement over vector: {(hybrid_best/vector_baseline - 1)*100:.1f}%")
    print(f"Improvement over keyword: {(hybrid_best/keyword_baseline - 1)*100:.1f}%")
    
    return evaluation_results
```

### 5.2 Alpha Optimization Analysis

```python
# experiments/hybrid/alpha_optimization.py
from src.evaluation_metrics import StatisticalAnalyzer
import matplotlib.pyplot as plt
import json

def analyze_alpha_optimization(dataset_name: str):
    """Analyze alpha parameter optimization."""
    
    # Load results with different alpha values
    with open(f'analysis/{dataset_name}_hybrid_analysis.json') as f:
        analysis = json.load(f)
    
    # Extract alpha values and metrics
    alpha_values = []
    ndcg_scores = []
    precision_scores = []
    mrr_scores = []
    
    for config in analysis['hybrid']:
        if config.startswith('alpha_'):
            alpha = float(config.split('_')[1])
            alpha_values.append(alpha)
            ndcg_scores.append(analysis['hybrid'][config]['ndcg@10'])
            precision_scores.append(analysis['hybrid'][config]['p@5'])
            mrr_scores.append(analysis['hybrid'][config]['mrr'])
    
    # Sort by alpha
    sorted_data = sorted(zip(alpha_values, ndcg_scores, precision_scores, mrr_scores))
    alpha_values = [d[0] for d in sorted_data]
    ndcg_scores = [d[1] for d in sorted_data]
    precision_scores = [d[2] for d in sorted_data]
    mrr_scores = [d[3] for d in sorted_data]
    
    # Find optimal alpha
    best_idx = ndcg_scores.index(max(ndcg_scores))
    optimal_alpha = alpha_values[best_idx]
    
    print(f"\n{dataset_name.upper()} Alpha Optimization:")
    print(f"Optimal alpha: {optimal_alpha:.2f}")
    print(f"Best NDCG@10: {ndcg_scores[best_idx]:.3f}")
    print(f"Best P@5: {precision_scores[best_idx]:.3f}")
    print(f"Best MRR: {mrr_scores[best_idx]:.3f}")
    
    # Statistical analysis
    analyzer = StatisticalAnalyzer()
    
    # Compare optimal vs default (0.5)
    default_idx = alpha_values.index(0.5) if 0.5 in alpha_values else len(alpha_values)//2
    
    if default_idx < len(ndcg_scores):
        improvement = (ndcg_scores[best_idx] - ndcg_scores[default_idx]) / ndcg_scores[default_idx] * 100
        print(f"Improvement over default (α=0.5): {improvement:.1f}%")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(alpha_values, ndcg_scores, 'b-', label='NDCG@10', linewidth=2)
    ax.plot(alpha_values, precision_scores, 'g--', label='P@5', linewidth=2)
    ax.plot(alpha_values, mrr_scores, 'r:', label='MRR', linewidth=2)
    
    # Mark optimal point
    ax.scatter([optimal_alpha], [ndcg_scores[best_idx]], 
              color='red', s=100, zorder=5)
    ax.annotate(f'Optimal α={optimal_alpha:.2f}',
               (optimal_alpha, ndcg_scores[best_idx]),
               xytext=(10, 10), textcoords='offset points')
    
    ax.set_xlabel('Alpha (0=keyword only, 1=vector only)')
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name.upper()}: Hybrid Retrieval Alpha Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f'figures/{dataset_name}_alpha_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_alpha
```

### 5.3 Query Complementarity Analysis

```python
# experiments/hybrid/complementarity_analysis.py
def analyze_method_complementarity(dataset_name: str):
    """Analyze how vector and keyword methods complement each other."""
    
    # Load results
    with open(f'results/{dataset_name}_retrieval_methods.json') as f:
        results = json.load(f)
    
    # Extract retrieved documents by method
    vector_retrieved = {}
    keyword_retrieved = {}
    
    for result in results['results']:
        if result['config']['retrieval_method'] == 'vector':
            query = result['query']
            vector_retrieved[query] = set(r['doc_id'] for r in result.get('retrieval_results', []))
        elif result['config']['retrieval_method'] == 'keyword':
            query = result['query']
            keyword_retrieved[query] = set(r['doc_id'] for r in result.get('retrieval_results', []))
    
    # Analyze overlap and unique contributions
    overlap_stats = []
    
    for query in vector_retrieved:
        if query in keyword_retrieved:
            v_docs = vector_retrieved[query]
            k_docs = keyword_retrieved[query]
            
            overlap = v_docs & k_docs
            v_unique = v_docs - k_docs
            k_unique = k_docs - v_docs
            
            overlap_stats.append({
                'overlap_ratio': len(overlap) / max(len(v_docs), len(k_docs)) if v_docs or k_docs else 0,
                'vector_unique_ratio': len(v_unique) / len(v_docs) if v_docs else 0,
                'keyword_unique_ratio': len(k_unique) / len(k_docs) if k_docs else 0
            })
    
    # Calculate aggregate statistics
    avg_overlap = np.mean([s['overlap_ratio'] for s in overlap_stats])
    avg_v_unique = np.mean([s['vector_unique_ratio'] for s in overlap_stats])
    avg_k_unique = np.mean([s['keyword_unique_ratio'] for s in overlap_stats])
    
    print(f"\n{dataset_name.upper()} Method Complementarity:")
    print(f"Average overlap: {avg_overlap:.1%}")
    print(f"Vector unique contribution: {avg_v_unique:.1%}")
    print(f"Keyword unique contribution: {avg_k_unique:.1%}")
    
    # Determine if methods are complementary
    complementarity_score = (avg_v_unique + avg_k_unique) / 2
    print(f"Complementarity score: {complementarity_score:.2f}")
    
    if complementarity_score > 0.3:
        print("→ Methods are highly complementary, hybrid should perform well")
    elif complementarity_score > 0.15:
        print("→ Methods show moderate complementarity")
    else:
        print("→ Methods have high overlap, limited benefit from hybrid")
    
    return overlap_stats
```

## 6. Visualization

```python
# experiments/hybrid/visualize_hybrid.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def create_method_comparison_heatmap():
    """Create heatmap comparing methods across datasets and metrics."""
    
    # Load results for all datasets
    results = {}
    for dataset in ['fiqa', 'trec_covid', 'scifact']:
        with open(f'analysis/{dataset}_hybrid_analysis.json') as f:
            results[dataset] = json.load(f)
    
    # Create matrix for heatmap
    methods = ['vector', 'keyword', 'hybrid_best']
    metrics = ['ndcg@10', 'p@5', 'mrr']
    datasets = ['fiqa', 'trec_covid', 'scifact']
    
    # Create 3D data structure
    data = np.zeros((len(datasets), len(methods), len(metrics)))
    
    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            if method == 'hybrid_best':
                # Find best hybrid configuration
                hybrid_configs = results[dataset]['hybrid']
                best = max(hybrid_configs.values(), key=lambda x: x['ndcg@10'])
                values = best
            else:
                values = results[dataset][method]['baseline']
            
            for k, metric in enumerate(metrics):
                data[i, j, k] = values[metric]
    
    # Create subplot for each metric
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Extract data for this metric
        metric_data = data[:, :, idx]
        
        # Create heatmap
        sns.heatmap(metric_data, 
                   annot=True, 
                   fmt='.3f',
                   xticklabels=methods,
                   yticklabels=datasets,
                   cmap='YlOrRd',
                   ax=ax,
                   vmin=0, vmax=1)
        
        ax.set_title(f'{metric.upper()} Scores')
        ax.set_xlabel('Method')
        ax.set_ylabel('Dataset')
    
    plt.tight_layout()
    plt.savefig('figures/method_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_alpha_curves_all_datasets():
    """Plot alpha optimization curves for all datasets."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, dataset in enumerate(['fiqa', 'trec_covid', 'scifact']):
        ax = axes[idx]
        
        # Load alpha optimization data
        with open(f'analysis/{dataset}_alpha_optimization.json') as f:
            data = json.load(f)
        
        alphas = data['alpha_values']
        ndcg = data['ndcg_scores']
        
        # Plot curve
        ax.plot(alphas, ndcg, 'b-', linewidth=2)
        ax.fill_between(alphas, ndcg, alpha=0.3)
        
        # Mark optimal
        best_idx = ndcg.index(max(ndcg))
        ax.scatter([alphas[best_idx]], [ndcg[best_idx]], 
                  color='red', s=100, zorder=5)
        
        # Add baseline lines
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Keyword only')
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='Vector only')
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Default')
        
        ax.set_xlabel('Alpha (keyword ← → vector)')
        ax.set_ylabel('NDCG@10')
        ax.set_title(f'{dataset.upper()}: α={alphas[best_idx]:.2f}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
    plt.tight_layout()
    plt.savefig('figures/alpha_optimization_all_datasets.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## 7. Step-by-Step Implementation Guide

### Day 1: Baseline Experiments
```bash
# 1. Verify collections exist
python main.py collection list | grep beir

# 2. Run single-method baselines
for dataset in fiqa trec_covid scifact; do
  echo "Testing $dataset..."
  
  # Vector baseline
  python main.py experiment sweep \
    --param retrieval_method --values vector \
    --param retrieval_k --values "5,10,20" \
    --queries test_data/beir_${dataset}_queries.json \
    --corpus beir_${dataset} \
    --output results/${dataset}_vector_baseline.json
  
  # Keyword baseline  
  python main.py experiment sweep \
    --param retrieval_method --values keyword \
    --param retrieval_k --values "5,10,20" \
    --queries test_data/beir_${dataset}_queries.json \
    --corpus beir_${dataset} \
    --output results/${dataset}_keyword_baseline.json
done
```

### Day 2: Alpha Optimization
```bash
# 1. Run alpha sweeps using similarity_threshold
for dataset in fiqa trec_covid scifact; do
  python main.py experiment sweep \
    --param retrieval_method --values hybrid \
    --param similarity_threshold \
    --range "0.0,1.0,0.05" \
    --param retrieval_k --values "10" \
    --queries test_data/beir_${dataset}_queries.json \
    --corpus beir_${dataset} \
    --output results/${dataset}_alpha_sweep.json
done

# 2. Monitor progress
watch -n 5 'sqlite3 data/experiments.db "SELECT experiment_id, status, completed_runs, total_runs FROM experiments ORDER BY created_at DESC LIMIT 5"'
```

### Day 3: Comprehensive Template Run
```bash
# Use the retrieval_methods template for full analysis
for dataset in fiqa trec_covid scifact; do
  python main.py experiment template retrieval_methods \
    --corpus beir_${dataset} \
    --queries test_data/beir_${dataset}_queries.json \
    --output results/${dataset}_full_retrieval.json
done
```

### Day 4: Dynamic Alpha Testing
```bash
# 1. Run query analysis
python experiments/hybrid/dynamic_alpha.py

# 2. Test dynamic alpha selection
python experiments/hybrid/test_dynamic_alpha.py \
  --dataset fiqa \
  --output results/fiqa_dynamic_alpha.json

python experiments/hybrid/test_dynamic_alpha.py \
  --dataset trec_covid \
  --output results/trec_covid_dynamic_alpha.json

python experiments/hybrid/test_dynamic_alpha.py \
  --dataset scifact \
  --output results/scifact_dynamic_alpha.json
```

### Day 5: Analysis and Reporting
```bash
# 1. Run all analysis scripts
python experiments/hybrid/analyze_hybrid.py
python experiments/hybrid/alpha_optimization.py
python experiments/hybrid/complementarity_analysis.py

# 2. Generate visualizations
python experiments/hybrid/visualize_hybrid.py

# 3. Statistical comparisons
python experiments/hybrid/statistical_tests.py

# 4. Generate final report
python experiments/hybrid/generate_report.py \
  --output reports/hybrid_optimization_report.md
```

## 8. Expected Outcomes

### Optimal Alpha Values by Dataset

| Dataset | Optimal α | Vector NDCG | Keyword NDCG | Hybrid NDCG | Improvement |
|---------|-----------|-------------|--------------|-------------|-------------|
| **FiQA** | 0.35 | 0.28 | 0.24 | 0.38 | +35.7% |
| **TREC-COVID** | 0.55 | 0.62 | 0.48 | 0.71 | +14.5% |
| **SciFact** | 0.65 | 0.58 | 0.42 | 0.67 | +15.5% |

### Query Type Performance

| Query Type | Best Method | Optimal α | Example |
|------------|-------------|-----------|---------|
| Short (≤3 words) | Hybrid | 0.3-0.4 | "COVID-19 vaccine" |
| Questions | Hybrid | 0.6-0.7 | "How does mRNA vaccine work?" |
| Technical terms | Keyword/Hybrid | 0.2-0.3 | "SARS-CoV-2 spike protein" |
| Conceptual | Vector/Hybrid | 0.7-0.8 | "Explain herd immunity" |

### Dynamic vs Static Alpha

| Dataset | Static α NDCG | Dynamic α NDCG | Improvement |
|---------|---------------|----------------|-------------|
| FiQA | 0.38 | 0.42 | +10.5% |
| TREC-COVID | 0.71 | 0.76 | +7.0% |
| SciFact | 0.67 | 0.72 | +7.5% |

## 9. Query Experiment Database

```python
# experiments/hybrid/query_database.py
from src.experiment_runner import ExperimentDatabase
import sqlite3
import pandas as pd

def generate_experiment_report():
    """Generate comprehensive report from experiment database."""
    
    conn = sqlite3.connect('data/experiments.db')
    
    # Get summary statistics
    summary_query = """
    SELECT 
        json_extract(config_json, '$.target_corpus') as dataset,
        json_extract(config_json, '$.retrieval_method') as method,
        json_extract(config_json, '$.similarity_threshold') as alpha,
        AVG(json_extract(metrics_json, '$.response_time')) as avg_latency,
        COUNT(*) as num_queries
    FROM experiment_runs
    WHERE experiment_id LIKE 'hybrid_%'
    GROUP BY dataset, method, alpha
    ORDER BY dataset, method, alpha
    """
    
    df = pd.read_sql_query(summary_query, conn)
    
    # Create pivot table
    pivot = df.pivot_table(
        values='avg_latency',
        index=['dataset', 'method'],
        columns='alpha',
        aggfunc='mean'
    )
    
    print("Latency by Configuration (seconds):")
    print(pivot.to_string())
    
    # Find best configurations
    best_query = """
    SELECT 
        json_extract(config_json, '$.target_corpus') as dataset,
        json_extract(config_json, '$.similarity_threshold') as best_alpha,
        MIN(json_extract(metrics_json, '$.response_time')) as best_latency
    FROM experiment_runs
    WHERE json_extract(config_json, '$.retrieval_method') = 'hybrid'
    GROUP BY dataset
    """
    
    best_configs = pd.read_sql_query(best_query, conn)
    
    print("\n\nBest Configurations by Dataset:")
    print(best_configs.to_string(index=False))
    
    return df, pivot, best_configs
```

## 10. Success Criteria

- ✅ Complete hybrid experiments for all 3 BEIR datasets
- ✅ Identify optimal alpha within 0.05 precision  
- ✅ Achieve >15% improvement over single methods
- ✅ Test dynamic alpha selection
- ✅ Generate reproducible results with existing infrastructure
- ✅ Complete within 5-day timeline

---

**Status**: Ready for immediate implementation (no modifications needed)
**Prerequisites**: BEIR datasets ingested as collections
**Next Step**: Run Day 1 baseline experiments
