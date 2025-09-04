# Research Proposal 3: Hybrid Retrieval Strategy Optimization for Query-Adaptive Performance

## 1. Executive Summary

**Research Goal**: Optimize the combination of dense vector and sparse keyword retrieval in a hybrid system to maximize retrieval effectiveness across diverse query types.

**Expected Impact**: 15-20% improvement in NDCG across mixed query sets with dynamic alpha adjustment.

**Timeline**: 5 days (40 hours total effort)

**Feasibility**: ⭐⭐⭐⭐⭐ Excellent - Hybrid retrieval already implemented in system

## 2. Research Question & Hypothesis

### Primary Research Question
How can dense vector and sparse keyword retrieval be optimally combined in a hybrid system to maximize retrieval effectiveness across diverse query types on consumer hardware?

### Secondary Questions
- What is the optimal alpha parameter for blending dense and sparse scores?
- Should alpha be static or dynamically adjusted based on query characteristics?
- How do different score normalization methods affect hybrid retrieval quality?

### Hypotheses
- **H1**: Hybrid retrieval will outperform single methods by >15% on mixed query sets
- **H2**: Dynamic alpha adjustment based on query type will improve MRR by >10% over static alpha
- **H3**: Reciprocal Rank Fusion will outperform linear combination for score aggregation

## 3. Literature Review & Gap Analysis

### Prior Work
- **Amazon (2024)**: Dense+sparse hybrid improved NDCG by 12-20% on product search
- **Pinecone (2024)**: Cascading retrieval unified dense/sparse with 10% recall improvement
- **Reddit RAG Community (2024)**: Reported 15% improvement with BM25+vectors

### Research Gap
No systematic study on optimal fusion strategies for small-scale RAG with both methods running on same hardware.

## 4. System Compatibility Analysis ✅

### Current System Support
- ✅ **Vector Retrieval**: Fully implemented with sqlite-vec
- ✅ **Keyword Retrieval**: FTS5 integration complete
- ✅ **Hybrid Mode**: Already available in retriever
- ✅ **similarity_threshold**: Can be used as alpha parameter
- ✅ **Template Available**: `retrieval_methods` template exists

### Required Modifications
- Minor: Add query analyzer for dynamic alpha selection (optional enhancement)

## 5. Preliminary Work

### 5.1 Query Analyzer Implementation (Optional Enhancement)

```python
# src/query_analyzer.py
import re
from typing import Tuple

class QueryAnalyzer:
    """Analyze queries to determine optimal retrieval strategy."""
    
    def __init__(self):
        self.keyword_indicators = {
            "version_patterns": r'\d+\.\d+',
            "error_codes": r'ERROR_?\w+|E\d{3,}',
            "technical_ids": r'[A-Z]{2,}_\w+',
            "proper_nouns": self._load_proper_nouns()
        }
    
    def analyze_query(self, query: str) -> Tuple[str, float]:
        """Determine optimal retrieval strategy and alpha."""
        
        # Check for keyword-dominant patterns
        has_version = bool(re.search(self.keyword_indicators["version_patterns"], query))
        has_error = bool(re.search(self.keyword_indicators["error_codes"], query))
        has_technical = bool(re.search(self.keyword_indicators["technical_ids"], query))
        
        # Calculate semantic complexity
        word_count = len(query.split())
        has_question_words = any(w in query.lower() for w in ['how', 'why', 'what', 'explain'])
        
        # Determine strategy and alpha
        if has_version or has_error or has_technical:
            return "keyword_dominant", 0.3  # More keyword weight
        elif has_question_words and word_count > 5:
            return "semantic_dominant", 0.7  # More vector weight
        else:
            return "balanced", 0.5
    
    def _load_proper_nouns(self):
        """Load list of technical proper nouns."""
        return ["Python", "JavaScript", "Docker", "Kubernetes", "MongoDB", "PostgreSQL"]
```

### 5.2 Test Query Development

Create `test_data/hybrid_queries.json`:

```json
{
  "categories": {
    "keyword_optimal": [
      {"query": "Python 3.9.7 compatibility issues", "optimal_alpha": 0.2},
      {"query": "org.springframework.boot version 2.5.0", "optimal_alpha": 0.2},
      {"query": "ERROR_CODE_404 meaning", "optimal_alpha": 0.1},
      {"query": "MacOS Monterey 12.3.1 requirements", "optimal_alpha": 0.2},
      {"query": "TCP port 8080 configuration", "optimal_alpha": 0.3}
    ],
    "vector_optimal": [
      {"query": "How does machine learning work?", "optimal_alpha": 0.8},
      {"query": "Explain the concept of recursion", "optimal_alpha": 0.8},
      {"query": "What are the benefits of cloud computing?", "optimal_alpha": 0.7},
      {"query": "Describe agile development methodology", "optimal_alpha": 0.7},
      {"query": "Understanding neural network architectures", "optimal_alpha": 0.8}
    ],
    "hybrid_optimal": [
      {"query": "How to fix Python ImportError in version 3.8?", "optimal_alpha": 0.5},
      {"query": "Explain Kubernetes pod scheduling algorithm", "optimal_alpha": 0.5},
      {"query": "Compare React 17 and React 18 features", "optimal_alpha": 0.5},
      {"query": "Debug MongoDB connection timeout issues", "optimal_alpha": 0.4},
      {"query": "REST API best practices in Node.js", "optimal_alpha": 0.6}
    ],
    "ambiguous": [
      {"query": "Java performance", "optimal_alpha": 0.6},
      {"query": "Python debugging", "optimal_alpha": 0.5},
      {"query": "Cloud migration", "optimal_alpha": 0.6},
      {"query": "API design", "optimal_alpha": 0.7},
      {"query": "Database optimization", "optimal_alpha": 0.5}
    ]
  }
}
```

### 5.3 Corpus Preparation

```bash
# Create diverse corpus for hybrid testing
mkdir -p corpus/hybrid/{technical_specs,conceptual,mixed,proper_nouns}

# Technical specifications (keyword-heavy)
# API docs, config files, error codes
python scripts/fetch_api_docs.py --count 1500 --output corpus/hybrid/technical_specs/

# Conceptual content (semantic-rich)
# Tutorials, explanations, guides
python scripts/fetch_tutorials.py --count 1500 --output corpus/hybrid/conceptual/

# Mixed content
# Blog posts with code, technical articles
python scripts/fetch_devto.py --count 1500 --output corpus/hybrid/mixed/

# Proper noun heavy content
# Product documentation, tool comparisons
python scripts/fetch_product_docs.py --count 500 --output corpus/hybrid/proper_nouns/

# Create collection and ingest
python main.py collection create hybrid_test --description "Hybrid retrieval evaluation"
python main.py ingest directory corpus/hybrid --collection hybrid_test --deduplicate

# Verify both indexes are built
python main.py analytics stats --collection hybrid_test
```

## 6. Experimental Design

### 6.1 Independent Variables

| Parameter | Values | Total |
|-----------|--------|-------|
| retrieval_method | [vector, keyword, hybrid] | 3 |
| alpha (similarity_threshold) | [0.0, 0.1, 0.2, ..., 1.0] | 11 |
| alpha_strategy | [static, dynamic] | 2 |
| retrieval_k | [5, 10, 15, 20] | 4 |
| query_type | [keyword_optimal, vector_optimal, hybrid_optimal, ambiguous] | 4 |

**Total Configurations**: 3 × 11 × 2 × 4 × 4 = 1056

### 6.2 Dependent Variables

- **Retrieval Effectiveness**: P@K, R@K, F1@K, MAP, MRR, NDCG
- **Coverage Metrics**: Unique docs retrieved, query coverage, domain coverage
- **Efficiency Metrics**: Query latency, memory usage, throughput
- **Robustness Metrics**: Failure rate, consistency, edge case performance

## 7. Experimental Procedure

### Phase 1: Single Method Baselines (Day 1)

```bash
# Test each retrieval method independently
for method in vector keyword; do
  python main.py experiment sweep \
    --param retrieval_method \
    --values $method \
    --queries test_data/hybrid_queries.json \
    --corpus hybrid_test \
    --output results/baseline_${method}.json
done

# Analyze method strengths
python experiments/hybrid/analyze_baselines.py \
  --vector results/baseline_vector.json \
  --keyword results/baseline_keyword.json \
  --output analysis/method_strengths.json

# Calculate complementarity
python experiments/hybrid/complementarity_analysis.py \
  --queries test_data/hybrid_queries.json \
  --corpus hybrid_test \
  --output analysis/method_overlap.json
```

### Phase 2: Static Alpha Optimization (Day 2)

```bash
# Use retrieval_methods template for comprehensive testing
python main.py experiment template retrieval_methods \
  --corpus hybrid_test \
  --queries test_data/hybrid_queries.json \
  --output results/retrieval_methods_full.json

# Fine-grained alpha sweep
python main.py experiment sweep \
  --param similarity_threshold \
  --range "0.0,1.0,0.05" \
  --queries test_data/hybrid_queries.json \
  --corpus hybrid_test \
  --output results/alpha_sweep_fine.json

# Query-type specific alpha optimization
for qtype in keyword_optimal vector_optimal hybrid_optimal ambiguous; do
  python experiments/hybrid/optimize_alpha.py \
    --query-type $qtype \
    --queries test_data/${qtype}_queries.json \
    --output results/alpha_${qtype}.json
done

# Find global optimal alpha
python experiments/hybrid/find_optimal_alpha.py \
  --results results/alpha_sweep_fine.json \
  --metric ndcg@10 \
  --output config/optimal_alpha.json
```

### Phase 3: Dynamic Alpha Strategy (Day 3)

```bash
# Implement query analyzer
cp experiments/hybrid/query_analyzer.py src/

# Test dynamic alpha selection
python experiments/hybrid/test_dynamic_alpha.py \
  --analyzer rule_based \
  --queries test_data/hybrid_queries.json \
  --corpus hybrid_test \
  --output results/dynamic_alpha_rules.json

# Train ML-based alpha predictor
python experiments/hybrid/train_alpha_predictor.py \
  --training-queries test_data/training_queries.json \
  --validation-queries test_data/validation_queries.json \
  --output models/alpha_predictor.pkl

# Test ML-based dynamic alpha
python experiments/hybrid/test_dynamic_alpha.py \
  --analyzer ml_based \
  --model models/alpha_predictor.pkl \
  --queries test_data/hybrid_queries.json \
  --output results/dynamic_alpha_ml.json

# Compare static vs dynamic
python main.py experiment compare \
  --config-a config/static_alpha.json \
  --config-b config/dynamic_alpha.json \
  --queries test_data/hybrid_queries.json \
  --significance 0.05 \
  --output results/static_vs_dynamic.json
```

### Phase 4: Advanced Fusion Methods (Day 4)

```bash
# Test different fusion methods
python experiments/hybrid/test_fusion_methods.py \
  --methods "linear,rrf,weighted_rrf,cascading" \
  --queries test_data/hybrid_queries.json \
  --corpus hybrid_test \
  --output results/fusion_comparison.json

# Score normalization comparison
python experiments/hybrid/test_normalization.py \
  --methods "min_max,z_score,rank_based,none" \
  --queries test_data/hybrid_queries.json \
  --output results/normalization_comparison.json

# Cascading retrieval strategies
python experiments/hybrid/test_cascading.py \
  --strategies experiments/hybrid/cascade_strategies.json \
  --queries test_data/hybrid_queries.json \
  --output results/cascading_results.json
```

### Phase 5: Validation and Production Config (Day 5)

```bash
# Cross-validation on external dataset
python experiments/hybrid/cross_validate.py \
  --config config/optimal_hybrid.json \
  --dataset data/ms_marco_subset.json \
  --output validation/external_validation.json

# Performance profiling
python experiments/hybrid/profile_performance.py \
  --config config/optimal_hybrid.json \
  --queries test_data/benchmark_queries.json \
  --iterations 100 \
  --output profiling/hybrid_performance.json

# Edge case testing
python experiments/hybrid/test_edge_cases.py \
  --queries test_data/edge_cases.json \
  --output validation/edge_case_results.json

# Generate final configuration
python experiments/hybrid/generate_production_config.py \
  --optimal results/optimal_configurations.json \
  --constraints config/production_constraints.yaml \
  --output config/hybrid_production.yaml
```

## 8. Implementation Guide

### Day 0: Setup and Verification
```bash
# 1. Verify hybrid retrieval works
python -c "
from src.retriever import Retriever
r = Retriever('data/rag_vectors.db')
print('Vector:', r.retrieve('test', method='vector', k=3))
print('Keyword:', r.retrieve('test', method='keyword', k=3))
print('Hybrid:', r.retrieve('test', method='hybrid', k=3))
"

# 2. Create experiment structure
mkdir -p experiments/hybrid/{scripts,analysis,results,config}

# 3. Install dependencies
pip install scipy numpy pandas matplotlib seaborn scikit-learn

# 4. Verify template availability
python main.py experiment template --list-templates | grep retrieval_methods
```

### Day 1: Baselines and Corpus
```bash
# 1. Prepare diverse corpus (5000 documents)
bash experiments/hybrid/prepare_corpus.sh

# 2. Create and populate collection
python main.py collection create hybrid_test
python main.py ingest directory corpus/hybrid --collection hybrid_test

# 3. Verify both indexes
python main.py maintenance validate --collection hybrid_test

# 4. Run baseline experiments
for method in vector keyword; do
  python main.py query "machine learning" \
    --collection hybrid_test \
    --k 10 \
    --metrics
done
```

### Day 2: Alpha Optimization
```bash
# 1. Run template experiment
python main.py experiment template retrieval_methods \
  --corpus hybrid_test \
  --queries test_data/hybrid_queries.json \
  --output results/methods_template.json

# 2. Alpha parameter sweep
python main.py experiment sweep \
  --param similarity_threshold \
  --values "0.0,0.2,0.4,0.5,0.6,0.8,1.0" \
  --queries test_data/hybrid_queries.json \
  --corpus hybrid_test \
  --output results/alpha_sweep.json

# 3. Analyze results
python experiments/hybrid/analyze_alpha.py \
  --results results/alpha_sweep.json \
  --output analysis/alpha_curves.json

# 4. Generate visualizations
python experiments/hybrid/visualize_alpha.py \
  --data analysis/alpha_curves.json \
  --output figures/alpha_optimization.png
```

### Day 3: Dynamic Alpha
```bash
# 1. Implement query analyzer
cp experiments/hybrid/query_analyzer.py src/

# 2. Test rule-based dynamic alpha
python experiments/hybrid/test_dynamic.py \
  --strategy rule_based \
  --output results/dynamic_rules.json

# 3. Train ML predictor
python experiments/hybrid/train_predictor.py \
  --output models/alpha_predictor.pkl

# 4. Compare strategies
python experiments/hybrid/compare_strategies.py \
  --static results/alpha_sweep.json \
  --dynamic results/dynamic_rules.json \
  --output analysis/strategy_comparison.json
```

### Day 4: Advanced Methods
```bash
# 1. Test fusion methods
python experiments/hybrid/test_fusion.py \
  --output results/fusion_methods.json

# 2. Normalization comparison
python experiments/hybrid/test_normalization.py \
  --output results/normalization.json

# 3. Cascading strategies
python experiments/hybrid/test_cascading.py \
  --output results/cascading.json

# 4. Find best combination
python experiments/hybrid/find_best_config.py \
  --results results/ \
  --output config/best_hybrid.json
```

### Day 5: Validation and Deployment
```bash
# 1. External validation
python experiments/hybrid/validate_external.py \
  --config config/best_hybrid.json \
  --output validation/external.json

# 2. Performance testing
python experiments/hybrid/benchmark.py \
  --config config/best_hybrid.json \
  --iterations 1000 \
  --output benchmark/performance.json

# 3. Generate report
python experiments/hybrid/generate_report.py \
  --results results/ \
  --analysis analysis/ \
  --validation validation/ \
  --output reports/hybrid_optimization.md

# 4. Update production config
python main.py config set retrieval_method hybrid
python main.py config set similarity_threshold 0.5
```

## 9. Validity Checks

### 9.1 Method Complementarity
```python
def validate_complementarity():
    """Ensure methods retrieve different relevant documents."""
    for query in test_queries:
        vector_docs = set(vector_retrieve(query, k=20))
        keyword_docs = set(keyword_retrieve(query, k=20))
        
        overlap = vector_docs & keyword_docs
        unique_vector = vector_docs - keyword_docs
        unique_keyword = keyword_docs - vector_docs
        
        # Should have reasonable uniqueness
        assert len(unique_vector) > 0.2 * len(vector_docs)
        assert len(unique_keyword) > 0.2 * len(keyword_docs)
        
        # But also some overlap
        assert len(overlap) > 0.3 * min(len(vector_docs), len(keyword_docs))
```

### 9.2 Score Distribution Analysis
```python
def analyze_score_distributions():
    """Validate score normalization effectiveness."""
    vector_scores = collect_scores(method="vector")
    keyword_scores = collect_scores(method="keyword")
    
    # Check distribution characteristics
    print(f"Vector: μ={np.mean(vector_scores):.3f}, σ={np.std(vector_scores):.3f}")
    print(f"Keyword: μ={np.mean(keyword_scores):.3f}, σ={np.std(keyword_scores):.3f}")
    
    # Test normalization methods
    for method in ["min_max", "z_score", "rank"]:
        normalized_v = normalize(vector_scores, method)
        normalized_k = normalize(keyword_scores, method)
        
        # Should have similar ranges after normalization
        assert abs(np.mean(normalized_v) - np.mean(normalized_k)) < 0.1
```

## 10. Expected Outcomes

### Performance Improvements
- **NDCG@10**: 15-20% improvement with hybrid
- **MRR**: 12-18% improvement
- **Query Coverage**: 95%+ with relevant results

### Optimal Configurations
- **Global Static Alpha**: 0.5-0.6 expected
- **Query-Specific Alphas**:
  - Keyword queries: 0.2-0.3
  - Semantic queries: 0.7-0.8
  - Mixed queries: 0.4-0.6

### Method Performance
- **Vector**: Best for conceptual queries
- **Keyword**: Best for specific terms
- **Hybrid**: Best overall performance

## 11. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|---------|------------|
| Score incompatibility | Low | Medium | Implement normalization |
| Increased latency | Medium | Low | Optimize parallel execution |
| Complex configuration | Low | Low | Provide clear defaults |

## 12. Success Criteria

- ✅ Demonstrate 15%+ NDCG improvement with hybrid
- ✅ Find optimal static alpha within 0.05 precision
- ✅ Show 10%+ improvement with dynamic alpha
- ✅ Complete 1000+ experimental configurations
- ✅ Achieve <300ms query latency

## 13. Resource Requirements

### Hardware
- Mac mini M4 with 16GB RAM
- 50GB storage for corpus and results

### Software
- Existing RAG system with hybrid support
- Python scientific stack

### Time
- 40 person-hours over 5 days
- ~15 compute-hours for experiments

## 14. Deliverables

1. **Optimal Alpha Guidelines**: Static and dynamic configurations
2. **Query Analyzer**: Production-ready query classification
3. **Fusion Method Analysis**: Comparison of aggregation strategies
4. **Performance Report**: Complete experimental results
5. **Production Config**: Ready-to-deploy settings

## 15. Bibliography

1. Amazon. (2024). "Integrate Sparse and Dense Vectors in RAG."
2. Pinecone. (2024). "Cascading Retrieval: Unifying Dense and Sparse."
3. Lin et al. (2021). "A Few Brief Notes on DeepImpact and FiD."
4. Ma et al. (2024). "Hybrid Retrieval in Production Search Systems."

---

**Status**: Ready for immediate implementation
**Prerequisites**: ✅ All met - hybrid retrieval fully supported
**Next Step**: Execute Day 0 setup and verification
