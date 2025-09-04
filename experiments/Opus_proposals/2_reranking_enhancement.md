# Research Proposal 2: Two-Stage Retrieval with Cross-Encoder Reranking for Precision Enhancement

## 1. Executive Summary

**Research Goal**: Implement and evaluate lightweight cross-encoder reranking to significantly improve retrieval precision in resource-constrained RAG systems.

**Expected Impact**: 20-30% improvement in P@5 with acceptable latency trade-offs (<500ms reranking overhead).

**Timeline**: 5 days (40 hours total effort)

**Feasibility**: ⭐⭐⭐⭐ Very Good - Reranking parameters exist in config, requires model integration

## 2. Research Question & Hypothesis

### Primary Research Question
Can a lightweight cross-encoder reranking stage significantly improve retrieval precision in resource-constrained RAG systems without prohibitive latency penalties?

### Secondary Questions
- What is the optimal retrieve-then-rerank configuration (retrieve K, rerank to N)?
- Which cross-encoder models provide the best precision/speed trade-off on consumer hardware?
- How does reranking impact different query types (factual vs. analytical)?

### Hypotheses
- **H1**: Cross-encoder reranking will improve P@5 by >20% compared to single-stage retrieval
- **H2**: Optimal configuration is retrieve-50-rerank-to-5 for balanced precision/latency
- **H3**: Reranking provides greater benefits for ambiguous queries (>30%) than specific queries (<15%)

## 3. Literature Review & Gap Analysis

### Prior Work
- **Pinecone (2024)**: Cascading retrieval improved Recall@5 by 6.8% in production
- **Haystack (2024)**: NVIDIA NeMo reranker showed 23% NDCG improvement
- **Microsoft (2023)**: MS MARCO cross-encoders achieved 0.39 MRR@10

### Research Gap
No studies on reranking effectiveness with quantized 4B models on consumer GPUs with memory constraints.

## 4. System Compatibility Analysis ⚠️

### Current System Support
- ✅ **rerank_model**: Parameter exists in ExperimentConfig
- ✅ **rerank_top_k**: Parameter exists in ExperimentConfig
- ⚠️ **Reranker Integration**: Not currently implemented in pipeline
- ✅ **Experiment Framework**: Ready for testing once integrated

### Required Modifications
```python
# src/reranker_service.py (NEW - needs creation)
from sentence_transformers import CrossEncoder
import torch

class RerankerService:
    def __init__(self, model_name: str):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = CrossEncoder(model_name, device=self.device)
    
    def rerank(self, query: str, passages: List[str], top_k: int = 5):
        pairs = [[query, passage] for passage in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)
        results = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return results[:top_k]
```

## 5. Preliminary Work

### 5.1 Reranker Model Setup

```bash
# Install cross-encoder models
pip install sentence-transformers torch

# Download and test reranker models
python -c "
from sentence_transformers import CrossEncoder
models = [
    'cross-encoder/ms-marco-MiniLM-L-6-v2',  # 80MB, fast
    'BAAI/bge-reranker-base',                # 300MB, good
    'cross-encoder/ms-marco-MiniLM-L-12-v2'  # 140MB, balanced
]
for model in models:
    print(f'Loading {model}...')
    ce = CrossEncoder(model)
    print(f'✓ {model} loaded successfully')
"

# Create reranker integration module
cat > src/reranker_service.py << 'EOF'
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
import torch
import logging

logger = logging.getLogger(__name__)

class RerankerService:
    """Cross-encoder reranking service for two-stage retrieval."""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = None
        self._lazy_load()
    
    def _lazy_load(self):
        """Lazy load the model on first use."""
        if self.model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
    
    def rerank(self, query: str, passages: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rerank passages based on relevance to query."""
        self._lazy_load()
        
        if not passages:
            return []
        
        # Prepare query-passage pairs
        pairs = [[query, passage] for passage in passages]
        
        # Get relevance scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Sort by score and return top-k
        results = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def rerank_with_metadata(self, query: str, results: List, top_k: int = 5):
        """Rerank retrieval results maintaining metadata."""
        if not results:
            return []
        
        passages = [r.content for r in results]
        reranked_passages = self.rerank(query, passages, top_k)
        
        # Map back to original results with new scores
        passage_to_result = {r.content: r for r in results}
        reranked_results = []
        
        for passage, score in reranked_passages:
            result = passage_to_result[passage]
            result.rerank_score = score
            reranked_results.append(result)
        
        return reranked_results
EOF

# Integrate into retriever
cat >> src/retriever.py << 'EOF'

def retrieve_and_rerank(self, query: str, k: int = 50, rerank_k: int = 5, 
                        reranker_model: str = None) -> List[RetrievalResult]:
    """Two-stage retrieval with reranking."""
    # First stage: retrieve more candidates
    candidates = self.retrieve(query, k=k, method='hybrid')
    
    if reranker_model and len(candidates) > rerank_k:
        # Second stage: rerank to get top results
        from .reranker_service import RerankerService
        reranker = RerankerService(reranker_model)
        reranked = reranker.rerank_with_metadata(query, candidates, top_k=rerank_k)
        return reranked
    
    return candidates[:rerank_k]
EOF
```

### 5.2 Test Query Development

Create `test_data/reranking_queries.json`:

```json
{
  "categories": {
    "ambiguous": [
      {"query": "What is Python?", "description": "Programming language or snake"},
      {"query": "Explain transformers", "description": "ML architecture or electrical"},
      {"query": "What is a kernel?", "description": "OS, ML, or agriculture"},
      {"query": "Define shells", "description": "Computing, marine, or military"},
      {"query": "What are containers?", "description": "Docker or physical"}
    ],
    "specific": [
      {"query": "What is the time complexity of heapsort?"},
      {"query": "Define the softmax activation function"},
      {"query": "What port does SSH use by default?"},
      {"query": "What is the Python GIL?"},
      {"query": "Explain TCP three-way handshake"}
    ],
    "multi_aspect": [
      {"query": "Compare REST and GraphQL APIs"},
      {"query": "Advantages and disadvantages of microservices"},
      {"query": "When to use SQL vs NoSQL databases"},
      {"query": "Monolithic vs microservices architecture"},
      {"query": "Synchronous vs asynchronous programming"}
    ],
    "contextual": [
      {"query": "How does it work?", "needs_context": true},
      {"query": "What are the benefits?", "needs_context": true},
      {"query": "Explain the implementation", "needs_context": true},
      {"query": "What are the limitations?", "needs_context": true},
      {"query": "How to optimize it?", "needs_context": true}
    ]
  }
}
```

### 5.3 Corpus with Relevance Gradients

```bash
# Create graded relevance corpus
mkdir -p corpus/reranking/{highly_relevant,partially_relevant,tangentially_relevant,distractors}

# Highly relevant: Direct answers (1000 docs)
# Download from technical documentation
wget -r -l 1 https://docs.python.org/3/tutorial/ -P corpus/reranking/highly_relevant/

# Partially relevant: Related content (2000 docs)
# Blog posts and discussions
python scripts/scrape_medium.py --tag "programming" --count 2000 \
  --output corpus/reranking/partially_relevant/

# Tangentially relevant: Same domain, different topics (2000 docs)
python scripts/fetch_arxiv.py --category cs.AI --count 2000 \
  --output corpus/reranking/tangentially_relevant/

# Distractors: Keyword matches but wrong context (1000 docs)
python scripts/generate_distractors.py \
  --keywords "python,kernel,transformer,shell" \
  --count 1000 \
  --output corpus/reranking/distractors/

# Create and populate collection
python main.py collection create reranking --description "Reranking evaluation corpus"
python main.py ingest directory corpus/reranking --collection reranking --deduplicate
```

## 6. Experimental Design

### 6.1 Independent Variables

| Parameter | Values | Total |
|-----------|--------|-------|
| reranker_model | [None, ms-marco-L6, bge-base, ms-marco-L12] | 4 |
| retrieval_k | [10, 20, 30, 50, 100] | 5 |
| rerank_top_n | [3, 5, 7, 10] | 4 |
| query_type | [ambiguous, specific, multi_aspect, contextual] | 4 |
| retrieval_method | [vector, hybrid] | 2 |

**Total Configurations**: 4 × 5 × 4 × 4 × 2 = 640

### 6.2 Dependent Variables

- **Precision Metrics**: P@1, P@3, P@5, P@10
- **Ranking Metrics**: MRR, NDCG@10, MAP
- **Performance Metrics**: Retrieval latency, reranking latency, total latency, memory usage
- **Quality Metrics**: Answer accuracy, completeness, citation precision

## 7. Experimental Procedure

### Phase 1: Baseline Without Reranking (Day 1)

```bash
# Test single-stage retrieval baselines
for method in vector hybrid; do
  for k in 5 10 20; do
    python main.py experiment sweep \
      --param retrieval_k \
      --values $k \
      --queries test_data/reranking_queries.json \
      --corpus reranking \
      --output results/baseline_${method}_k${k}.json
  done
done

# Analyze baseline performance
python experiments/reranking/analyze_baseline.py \
  --results results/baseline_*.json \
  --output analysis/baseline_metrics.json
```

### Phase 2: Reranker Model Comparison (Days 2-3)

```bash
# Test different reranker models
MODELS="cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base,cross-encoder/ms-marco-MiniLM-L-12-v2"

for model in $(echo $MODELS | tr ',' ' '); do
  model_name=$(echo $model | sed 's/\//_/g')
  
  python main.py experiment sweep \
    --param rerank_model \
    --values "$model" \
    --queries test_data/reranking_queries.json \
    --corpus reranking \
    --output results/rerank_${model_name}.json
done

# Memory and latency profiling
python experiments/reranking/profile_models.py \
  --models "$MODELS" \
  --queries test_data/reranking_queries.json \
  --output profiling/model_performance.json
```

### Phase 3: Optimal Configuration Search (Day 4)

```bash
# Grid search for retrieve-k and rerank-n
python experiments/reranking/grid_search.py \
  --retrieve_k "10,20,30,50,100" \
  --rerank_n "3,5,7,10" \
  --model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
  --queries test_data/reranking_queries.json \
  --output results/grid_search.json

# Query-type specific optimization
for qtype in ambiguous specific multi_aspect contextual; do
  python main.py experiment sweep \
    --param rerank_top_k \
    --values "3,5,7,10" \
    --queries test_data/${qtype}_queries.json \
    --corpus reranking \
    --output results/qtype_${qtype}.json
done
```

### Phase 4: Statistical Validation (Day 5)

```bash
# A/B test: with vs without reranking
python main.py experiment compare \
  --config-a config/no_reranking.json \
  --config-b config/with_reranking.json \
  --queries test_data/reranking_queries.json \
  --significance 0.05 \
  --output results/ab_test_reranking.json

# External validation on MS MARCO subset
python experiments/reranking/validate_msmarco.py \
  --model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
  --dataset data/msmarco_subset.json \
  --output validation/msmarco_results.json
```

## 8. Implementation Guide

### Day 0: Setup and Integration
```bash
# 1. Install dependencies
pip install sentence-transformers torch scipy pandas matplotlib

# 2. Create reranker service
cp experiments/reranking/reranker_service.py src/

# 3. Integrate with retriever
python experiments/reranking/integrate_reranker.py

# 4. Test integration
python -c "
from src.retriever import Retriever
from src.reranker_service import RerankerService
print('✓ Reranker integration successful')
"

# 5. Download all reranker models
python experiments/reranking/download_models.py
```

### Day 1: Corpus and Baseline
```bash
# 1. Prepare graded corpus (6000 documents)
bash experiments/reranking/prepare_corpus.sh

# 2. Ingest into collection
python main.py collection create reranking
python main.py ingest directory corpus/reranking --collection reranking

# 3. Verify corpus statistics
python main.py analytics stats --collection reranking

# 4. Run baseline experiments
python experiments/reranking/run_baseline.py
```

### Day 2-3: Model Experiments
```bash
# 1. Test MS MARCO models
for model in minilm-l6 minilm-l12; do
  python experiments/reranking/test_model.py \
    --model cross-encoder/ms-marco-${model}-v2 \
    --output results/marco_${model}.json
done

# 2. Test BGE rerankers
python experiments/reranking/test_model.py \
  --model BAAI/bge-reranker-base \
  --output results/bge_base.json

# 3. Profile performance
python experiments/reranking/profile.py --all-models
```

### Day 4: Optimization
```bash
# 1. Grid search
python experiments/reranking/optimize.py \
  --method grid_search \
  --output optimization/grid_results.json

# 2. Query-specific tuning
python experiments/reranking/tune_by_query_type.py

# 3. Find optimal configuration
python experiments/reranking/find_optimal.py \
  --metric ndcg@10 \
  --output config/optimal_reranking.json
```

### Day 5: Validation and Deployment
```bash
# 1. Statistical tests
python experiments/reranking/statistical_tests.py \
  --baseline results/baseline.json \
  --reranked results/optimized.json

# 2. Generate report
python experiments/reranking/generate_report.py \
  --template reports/reranking_template.md \
  --output reports/reranking_final.md

# 3. Export configuration
python main.py config set rerank_model "cross-encoder/ms-marco-MiniLM-L-6-v2"
python main.py config set rerank_top_k 5
```

## 9. Validity Checks

### 9.1 Ranking Quality Validation
```python
def validate_ranking_improvement():
    """Ensure reranking improves ranking metrics."""
    metrics_before = calculate_ranking_metrics(baseline_results)
    metrics_after = calculate_ranking_metrics(reranked_results)
    
    improvements = {
        "mrr_delta": metrics_after["mrr"] - metrics_before["mrr"],
        "ndcg_delta": metrics_after["ndcg"] - metrics_before["ndcg"],
        "map_delta": metrics_after["map"] - metrics_before["map"]
    }
    
    # All metrics should improve
    assert all(delta > 0 for delta in improvements.values())
    assert ttest_paired(metrics_before, metrics_after).pvalue < 0.05
```

### 9.2 Latency Constraints
```python
def validate_latency_constraints():
    """Ensure reranking stays within latency budget."""
    latency_budget_ms = 2000  # 2 second total
    
    timings = measure_latencies(reranking_pipeline)
    assert timings["total"] < latency_budget_ms
    assert timings["reranking"] < 500  # Max 500ms for reranking
    assert timings["reranking"] / timings["total"] < 0.4  # <40% of total
```

## 10. Expected Outcomes

### Performance Improvements
- **P@5**: 20-30% improvement with reranking
- **MRR**: 15-25% improvement
- **NDCG@10**: 18-28% improvement

### Latency Trade-offs
- **Retrieval**: 100-200ms (first stage)
- **Reranking**: 200-400ms (second stage)
- **Total**: <600ms end-to-end

### Model Recommendations
- **Fast**: ms-marco-MiniLM-L-6 (80MB, 200ms)
- **Balanced**: ms-marco-MiniLM-L-12 (140MB, 300ms)
- **Quality**: bge-reranker-base (300MB, 400ms)

## 11. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|---------|------------|
| Model integration issues | Medium | High | Test integration early |
| Memory constraints | Low | High | Monitor usage, use smaller models |
| Latency exceeds budget | Medium | Medium | Optimize batch processing |

## 12. Success Criteria

- ✅ Successfully integrate 3+ reranker models
- ✅ Achieve >20% improvement in P@5
- ✅ Maintain <500ms reranking latency
- ✅ Complete 640 experimental configurations
- ✅ Demonstrate statistical significance (p<0.05)

## 13. Resource Requirements

### Hardware
- Mac mini M4 with 16GB RAM
- MPS GPU acceleration
- 50GB storage for models and results

### Software
- sentence-transformers>=2.2.0
- torch with MPS support
- Existing RAG system

### Time
- 40 person-hours over 5 days
- ~20 compute-hours for experiments

## 14. Deliverables

1. **Reranker Integration**: Production-ready reranking service
2. **Experimental Results**: Complete performance data
3. **Model Comparison**: Detailed analysis of 3 reranker models
4. **Configuration Guide**: Optimal retrieve-k and rerank-n settings
5. **Research Report**: Publication-ready findings

## 15. Bibliography

1. Pinecone. (2024). "Cascading Retrieval: Unifying Dense and Sparse."
2. Haystack. (2024). "Optimize RAG with NVIDIA NeMo Reranking."
3. Thakur et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Retrieval."
4. Nogueira & Cho. (2019). "Passage Re-ranking with BERT."

---

**Status**: Ready for implementation after reranker integration
**Prerequisites**: ⚠️ Requires reranker service implementation
**Next Step**: Create reranker_service.py and integrate with pipeline
