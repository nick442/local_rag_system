# Research Proposal 1: Optimal Document Chunking Strategy for Resource-Constrained RAG Systems

## 1. Executive Summary

**Research Goal**: Determine optimal document chunking parameters for maximizing retrieval precision and answer quality on consumer hardware (Mac mini M4, 16GB RAM) using 4B parameter models.

**Expected Impact**: 15-25% improvement in retrieval precision with optimized chunking strategies while maintaining <2s query latency.

**Timeline**: 5 days (40 hours total effort)

**Feasibility**: ⭐⭐⭐⭐⭐ Excellent - System fully supports chunking experiments with existing infrastructure

## 2. Research Question & Hypothesis

### Primary Research Question
What is the optimal document chunking strategy for maximizing retrieval precision and answer quality in RAG systems deployed on consumer hardware (16GB RAM)?

### Secondary Questions
- How does chunk size affect the trade-off between semantic coherence and retrieval precision?
- What overlap ratio minimizes information loss while maintaining indexing efficiency?
- Do different document types (technical, narrative, reference) require different chunking strategies?

### Hypotheses
- **H1**: Smaller chunks (256-512 tokens) will yield higher retrieval precision (>0.75 P@5) but lower answer completeness
- **H2**: 25% overlap (128 tokens for 512-token chunks) provides optimal information continuity
- **H3**: Technical documents benefit from smaller chunks while narrative texts perform better with larger chunks

## 3. Literature Review & Gap Analysis

### Prior Work
- **Kurniawan (2024)**: Demonstrated 35% MRR improvement with 512-token chunks on cloud infrastructure
- **LlamaIndex (2024)**: Found optimal chunk size varies by embedding model (256 for MiniLM, 512 for MPNet)
- **AWS Research (2024)**: Showed diminishing returns beyond 30% overlap

### Research Gap
No systematic study on chunk optimization for 4B parameter models on consumer hardware with memory constraints.

## 4. System Compatibility Analysis ✅

### Current System Support
- ✅ **chunk_size**: Fully supported in ProfileConfig and ExperimentConfig
- ✅ **chunk_overlap**: Fully supported with configurable values
- ✅ **chunking_strategy**: Supported via experiment_templates.py
- ✅ **Template Available**: `chunk_optimization` template pre-defined
- ✅ **Experiment Runner**: Full parameter sweep support

### Required Modifications
- None - experiment can run with existing infrastructure

## 5. Preliminary Work

### 5.1 Corpus Preparation

**Target Corpus**: 5,000 documents across 3 categories

```bash
# Create corpus structure
mkdir -p corpus/{technical,narrative,reference}

# Download technical documentation (2000 docs)
# - ArXiv papers, Python docs, Stack Overflow answers
wget -r -l 2 -A "*.md,*.txt" https://docs.python.org/3/ -P corpus/technical/

# Download narrative content (2000 docs)
# - Wikipedia articles, technical blogs
python scripts/fetch_wikipedia.py --featured --count 2000 --output corpus/narrative/

# Download reference materials (1000 docs)
# - API docs, README files, FAQs
python scripts/github_crawler.py --stars ">1000" --count 1000 --output corpus/reference/

# Ingest corpus into collections
python main.py collection create technical --description "Technical documentation"
python main.py collection create narrative --description "Narrative texts"
python main.py collection create reference --description "Reference materials"

python main.py ingest directory corpus/technical --collection technical --deduplicate
python main.py ingest directory corpus/narrative --collection narrative --deduplicate
python main.py ingest directory corpus/reference --collection reference --deduplicate

# Verify corpus statistics
python main.py analytics stats --collection technical
python main.py analytics stats --collection narrative
python main.py analytics stats --collection reference
```

### 5.2 Query Set Development

Create evaluation queries in `test_data/chunking_queries.json`:

```json
{
  "categories": {
    "factual": [
      {"query": "What is the time complexity of quicksort?", "expected_chunks": 2},
      {"query": "Who invented the transformer architecture?", "expected_chunks": 1},
      {"query": "What year was Python released?", "expected_chunks": 1}
    ],
    "explanatory": [
      {"query": "How does backpropagation work?", "expected_chunks": 4},
      {"query": "Explain the attention mechanism", "expected_chunks": 5},
      {"query": "What are the principles of REST API design?", "expected_chunks": 3}
    ],
    "comparative": [
      {"query": "Compare Python and JavaScript for web development", "expected_chunks": 6},
      {"query": "Differences between CNN and RNN architectures", "expected_chunks": 5},
      {"query": "TCP vs UDP protocol comparison", "expected_chunks": 7}
    ],
    "procedural": [
      {"query": "How to implement a binary search tree?", "expected_chunks": 5},
      {"query": "Steps to deploy a Docker container", "expected_chunks": 4},
      {"query": "Process for training a neural network", "expected_chunks": 6}
    ]
  }
}
```

### 5.3 Dependencies & System Validation

```bash
# Verify system requirements
python main.py doctor --format markdown --output reports/system_health.md

# Check available templates
python main.py experiment template --list-templates

# Validate chunking setup
python -c "
from src.experiment_templates import get_template
template = get_template('chunk_optimization')
print(f'Template loaded: {template.name}')
print(f'Parameters: {[p.param_name for p in template.parameter_ranges]}')
"
```

## 6. Experimental Design

### 6.1 Independent Variables

| Parameter | Values | Total |
|-----------|--------|-------|
| chunk_size | [128, 256, 512, 768, 1024, 1536, 2048] | 7 |
| chunk_overlap | [0, 64, 128, 192, 256] | 5 |
| chunking_strategy | ["token", "sentence", "paragraph"] | 3 |
| document_type | ["technical", "narrative", "reference"] | 3 |

**Total Configurations**: 7 × 5 × 3 × 3 = 315

### 6.2 Dependent Variables

- **Retrieval Metrics**: P@K, R@K, MRR, NDCG@10
- **Answer Quality**: Faithfulness, relevancy, completeness, hallucination rate
- **Performance**: Indexing time, query latency, memory usage
- **Semantic**: Chunk coherence, distinctness, information density

### 6.3 Control Variables

```python
controlled_parameters = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "models/gemma-3-4b-it-q4_0.gguf",
    "retrieval_k": 5,
    "temperature": 0.3,
    "max_tokens": 1024,
    "similarity_metric": "cosine"
}
```

## 7. Experimental Procedure

### Phase 1: Baseline Establishment (Day 1)

```bash
# Establish baseline with current configuration
python main.py experiment sweep \
  --param chunk_size \
  --values 512 \
  --queries test_data/chunking_queries.json \
  --corpus technical \
  --output results/baseline_chunking.json

# Calculate baseline metrics
python experiments/chunking/analyze_baseline.py \
  --results results/baseline_chunking.json \
  --output metrics/baseline_metrics.json
```

### Phase 2: Grid Search (Days 2-3)

```bash
# Run comprehensive chunking optimization
python main.py experiment template chunk_optimization \
  --corpus technical \
  --queries test_data/chunking_queries.json \
  --output results/chunk_optimization_technical.json

# Run for each document type
for dtype in narrative reference; do
  python main.py experiment template chunk_optimization \
    --corpus $dtype \
    --queries test_data/chunking_queries.json \
    --output results/chunk_optimization_${dtype}.json
done

# Detailed parameter sweeps
python main.py experiment sweep \
  --param chunk_size \
  --values 128,256,512,768,1024,1536,2048 \
  --queries test_data/chunking_queries.json \
  --output results/chunk_size_sweep.json

python main.py experiment sweep \
  --param chunk_overlap \
  --values 0,32,64,128,192,256 \
  --queries test_data/chunking_queries.json \
  --output results/chunk_overlap_sweep.json
```

### Phase 3: Focused Optimization (Day 4)

```bash
# Identify top configurations
python experiments/chunking/find_optimal.py \
  --results results/ \
  --metric "precision@5" \
  --threshold 0.8 \
  --output analysis/top_configs.json

# Run repeated trials for statistical significance
for config in $(cat analysis/top_configs.json | jq -r '.configs[]'); do
  for run in {1..10}; do
    python main.py experiment sweep \
      --param chunk_size \
      --values $config \
      --queries test_data/chunking_queries.json \
      --output results/validation_${config}_${run}.json
  done
done
```

### Phase 4: Validation (Day 5)

```bash
# Cross-validation on external corpus
python main.py collection create validation --description "External validation set"
python main.py ingest directory corpus/validation --collection validation

python main.py experiment compare \
  --config-a balanced \
  --config-b analysis/optimal_config.json \
  --queries test_data/validation_queries.json \
  --significance 0.05 \
  --output results/validation_comparison.json

# Generate final report
python experiments/chunking/generate_report.py \
  --results results/ \
  --analysis analysis/ \
  --template reports/chunking_template.md \
  --output reports/chunking_optimization_report.md
```

## 8. Validity Checks

### 8.1 Statistical Validity

```python
# experiments/chunking/statistical_validation.py
from scipy import stats
import numpy as np

def validate_significance(results_a, results_b):
    """Ensure statistical significance of comparisons."""
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(results_a, results_b)
    
    # Effect size (Cohen's d)
    cohens_d = (np.mean(results_a) - np.mean(results_b)) / np.std(results_a - results_b)
    
    # Bootstrap confidence intervals
    ci_lower, ci_upper = bootstrap_ci(results_a - results_b, n_bootstrap=1000)
    
    return {
        "significant": p_value < 0.05,
        "effect_size": cohens_d,
        "confidence_interval": (ci_lower, ci_upper),
        "practical_significance": abs(cohens_d) > 0.5
    }
```

### 8.2 Construct Validity

- Ensure chunks maintain semantic coherence
- Verify sentence completion
- Check information preservation
- Measure chunk diversity

### 8.3 External Validity

- Test on unseen corpus
- Verify generalization across document types
- Compare with published benchmarks

## 9. Step-by-Step Implementation Guide

### Day 0: Environment Setup
```bash
# 1. Verify environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
python main.py doctor

# 2. Create experiment structure
mkdir -p experiments/chunking/{corpus,queries,results,analysis,scripts}

# 3. Install additional dependencies
pip install scipy matplotlib seaborn pandas

# 4. Verify templates
python main.py experiment template --list-templates | grep chunk
```

### Day 1: Corpus & Baseline
```bash
# 1. Prepare corpus (5000 documents)
bash experiments/chunking/prepare_corpus.sh

# 2. Create collections and ingest
python main.py collection create chunking_test
python main.py ingest directory corpus/ --collection chunking_test

# 3. Generate query sets
python experiments/chunking/generate_queries.py

# 4. Run baseline
python main.py query "test query" --collection chunking_test --metrics
```

### Day 2-3: Main Experiments
```bash
# 1. Run template experiment
python main.py experiment template chunk_optimization \
  --queries test_data/chunking_queries.json \
  --corpus chunking_test \
  --output results/chunk_main.json

# 2. Analyze results in real-time
tail -f results/chunk_main.json | python experiments/chunking/live_analysis.py
```

### Day 4: Analysis
```bash
# 1. Statistical analysis
python experiments/chunking/analyze_results.py \
  --input results/ \
  --output analysis/

# 2. Visualizations
python experiments/chunking/visualize.py \
  --data analysis/ \
  --output figures/
```

### Day 5: Validation & Report
```bash
# 1. External validation
python experiments/chunking/validate_external.py

# 2. Generate report
python experiments/chunking/report_generator.py \
  --output reports/final_chunking_report.md

# 3. Export optimal configuration
python main.py config set chunk_size 512
python main.py config set chunk_overlap 128
```

## 10. Expected Outcomes

### Performance Improvements
- **Retrieval Precision**: 15-25% improvement over baseline
- **Query Latency**: Maintain <2s end-to-end
- **Memory Usage**: Stay within 12GB operational limit

### Scientific Contributions
- First systematic chunking study for 4B models on consumer hardware
- Document-type specific chunking recommendations
- Overlap ratio optimization curves

### Practical Guidelines
- Optimal chunk sizes for different use cases
- Memory-performance trade-off curves
- Production-ready configuration files

## 11. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|---------|------------|
| Memory overflow | Low | High | Monitor usage, reduce batch size |
| Long runtime | Medium | Medium | Use subset for initial tests |
| Poor results | Low | High | Verify corpus quality first |

## 12. Success Criteria

- ✅ Complete 315 experimental configurations
- ✅ Achieve >15% improvement in P@5
- ✅ Maintain <2s query latency
- ✅ Generate reproducible results (CV < 10%)
- ✅ Produce actionable configuration recommendations

## 13. Resource Requirements

### Hardware
- Mac mini M4 with 16GB RAM
- 100GB storage for corpus and results

### Time
- 40 person-hours over 5 days
- ~15 compute-hours for experiments

### Data
- 5000 documents (technical, narrative, reference)
- 200 evaluation queries with ground truth

## 14. Deliverables

1. **Experimental Data**: All raw results in JSON format
2. **Analysis Report**: Statistical analysis and findings
3. **Visualizations**: Performance curves and heatmaps
4. **Configuration Files**: Optimal settings for production
5. **Research Paper**: 6-8 page report suitable for publication

## 15. Bibliography

1. Kurniawan, S. (2024). "RAG Chunk Size Experiment." Medium.
2. LlamaIndex Team. (2024). "Evaluating the Ideal Chunk Size for RAG Systems."
3. AWS Big Data Blog. (2024). "Integrate Sparse and Dense Vectors in RAG."
4. Liu et al. (2024). "Retrieval-Augmented Generation for Knowledge-Intensive Tasks."

---

**Status**: Ready for implementation
**Prerequisites**: ✅ All met
**Next Step**: Execute Day 0 setup commands
