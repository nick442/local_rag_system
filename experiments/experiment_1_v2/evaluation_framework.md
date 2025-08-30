# Evaluation Framework: Metrics Collection and Analysis for Chunking Experiments

## Overview
This document defines the comprehensive evaluation framework for Experiment 1 v2, addressing the insufficient metrics identified in the original experiment review. It establishes rigorous methodology for measuring retrieval quality, system performance, and statistical significance of chunking parameter optimization.

## Current Metrics Limitations (From Review)

### Original Experiment Issues
- ❌ **Only coarse metrics**: response_time, num_sources, response_length  
- ❌ **No retrieval precision/recall**: Can't evaluate retrieval quality
- ❌ **No token counts**: Missing prompt/context/generation token analysis
- ❌ **No timing breakdown**: Can't separate retrieval vs generation performance
- ❌ **No index metrics**: Missing chunk counts, storage impact analysis
- ❌ **Statistical invalidity**: n=3 per config, no confidence intervals

### New Framework Addresses
- ✅ **Rich retrieval metrics**: Precision@K, Recall@K, MRR, NDCG
- ✅ **Complete token analysis**: Prompt, context, generation token utilization  
- ✅ **Detailed timing**: Retrieval, generation, total time breakdown
- ✅ **Index analytics**: Storage size, chunk counts, memory usage per config
- ✅ **Statistical rigor**: 52 queries × 10 runs × 20 configs with confidence intervals

## Metrics Categories

### 1. Retrieval Quality Metrics

#### 1.1 Information Retrieval Standards
**Precision@K (P@K)**
- Definition: Fraction of top-K retrieved documents that are relevant
- Formula: P@K = (relevant docs in top-K) / K
- Measured at: K = 1, 3, 5
- Expected range: 0.6-0.9 for well-tuned systems
- Use case: Measuring retrieval accuracy

**Recall@K (R@K)**  
- Definition: Fraction of all relevant documents found in top-K results
- Formula: R@K = (relevant docs in top-K) / (total relevant docs)
- Measured at: K = 1, 3, 5
- Expected range: 0.3-0.7 for K=5
- Use case: Measuring retrieval completeness

**Mean Reciprocal Rank (MRR)**
- Definition: Average inverse rank of first relevant result
- Formula: MRR = (1/N) × Σ(1/rank_first_relevant)
- Range: 0.0-1.0 (higher is better)
- Expected range: 0.7-0.9 for good systems
- Use case: Measuring ranking quality

**Normalized Discounted Cumulative Gain@5 (NDCG@5)**
- Definition: Position-sensitive relevance measure with graded judgments
- Formula: NDCG@K = DCG@K / IDCG@K
- Range: 0.0-1.0 (higher is better)  
- Expected range: 0.8-0.95 for optimized systems
- Use case: Measuring ranking quality with relevance degrees

#### 1.2 Implementation Requirements
```python
class RetrievalQualityEvaluator:
    """Comprehensive retrieval quality evaluation."""
    
    def __init__(self, ground_truth_relevance: Dict[str, Dict[str, float]]):
        """
        Initialize with ground truth relevance judgments.
        Format: {query_id: {doc_id: relevance_score}}
        """
        self.ground_truth = ground_truth_relevance
    
    def calculate_precision_at_k(self, query_id: str, retrieved_docs: List[str], k: int) -> float:
        """Calculate Precision@K for a single query."""
        if not retrieved_docs or k <= 0:
            return 0.0
            
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k 
                               if self.ground_truth.get(query_id, {}).get(doc, 0) > 0)
        return relevant_in_top_k / k
    
    def calculate_recall_at_k(self, query_id: str, retrieved_docs: List[str], k: int) -> float:
        """Calculate Recall@K for a single query."""
        relevant_docs = [doc for doc, score in self.ground_truth.get(query_id, {}).items() 
                        if score > 0]
        if not relevant_docs:
            return 0.0
            
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_in_top_k / len(relevant_docs)
    
    def calculate_mrr(self, query_results: Dict[str, List[str]]) -> float:
        """Calculate Mean Reciprocal Rank across all queries."""
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
        """Calculate NDCG@K for a single query."""
        def dcg_at_k(relevance_scores: List[float], k: int) -> float:
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
```

### 2. System Performance Metrics

#### 2.1 Timing Analysis
**Retrieval Time**
- Definition: Time for database query + similarity search + result formatting
- Measurement: Start before retriever.retrieve(), end after contexts returned
- Expected range: 0.1-2.0 seconds for 10K+ document collections
- Factors: Index size, chunk count, similarity algorithm complexity

**Generation Time**  
- Definition: Time for LLM inference (prompt processing + token generation)
- Measurement: Start before LLM call, end after response completion
- Expected range: 2.0-8.0 seconds for 2048-token responses
- Factors: Model size, context length, generation tokens

**Total End-to-End Time**
- Definition: Complete query processing from input to response
- Measurement: Full query() method execution time
- Expected range: 3.0-10.0 seconds for typical queries
- Target: Minimize while maintaining quality

#### 2.2 Token Utilization Analysis
**Prompt Tokens**
- Definition: Tokens in system prompt + query + retrieved context
- Measurement: Token count before generation
- Expected range: 500-6000 tokens depending on chunk size/count
- Optimization target: Maximize information density

**Context Tokens**
- Definition: Tokens from retrieved document chunks only
- Measurement: Sum of tokens in retrieved contexts
- Expected range: 200-4000 tokens depending on chunking strategy
- Analysis: Context efficiency vs chunk parameters

**Generated Tokens**
- Definition: Tokens in LLM response output
- Measurement: Token count of generated response
- Expected range: 50-500 tokens for typical answers
- Quality indicator: Response completeness

**Token Utilization Efficiency**
- Definition: Generated tokens / Total prompt tokens
- Formula: Efficiency = output_tokens / (prompt_tokens + output_tokens)
- Expected range: 0.05-0.20 for efficient systems
- Optimization: Balance context vs generation token usage

#### 2.3 Resource Consumption Metrics
**Peak Memory Usage**
- Definition: Maximum RAM usage during query processing
- Measurement: Process memory monitoring during execution
- Expected range: 2-6 GB for M4 Mac with 16GB RAM
- Critical threshold: <12 GB to avoid system pressure

**Index Storage Size**
- Definition: Database size for each collection configuration
- Measurement: File size of collection-specific database/index
- Expected variation: 10-50% difference across chunk sizes
- Analysis: Storage efficiency vs retrieval performance

**Processing Overhead**
- Definition: Additional time/memory for experimental configuration
- Measurement: Config creation + collection setup time
- Expected range: 10-60 seconds per configuration
- Optimization: Minimize experimental overhead

### 3. Quality Consistency Metrics

#### 3.1 Response Quality Analysis
**BLEU Score Consistency**
- Definition: N-gram overlap similarity between responses across configs
- Purpose: Measure response content consistency across chunking parameters
- Expected range: 0.7-0.95 for consistent quality
- Analysis: Identify configs that change response quality

**Response Length Stability**
- Definition: Coefficient of variation in response word count
- Formula: CV = std(response_lengths) / mean(response_lengths)
- Expected range: 0.1-0.3 for stable systems
- Quality indicator: Consistent response completeness

**Semantic Similarity**
- Definition: Vector similarity between responses from different configs
- Measurement: Embedding-based cosine similarity
- Expected range: 0.8-0.95 for semantically consistent responses
- Analysis: Identify configs that change response meaning

#### 3.2 Retrieval Stability Analysis
**Document Set Overlap**
- Definition: Jaccard similarity of retrieved document sets across configs
- Formula: Jaccard = |A ∩ B| / |A ∪ B|
- Expected range: 0.6-0.9 for stable retrieval
- Analysis: How chunking affects document selection

**Ranking Stability**
- Definition: Spearman correlation of document rankings across configs
- Expected range: 0.7-0.9 for stable ranking
- Analysis: How chunking affects relevance ordering

## Ground Truth and Query Dataset

### Ground Truth Creation
**Manual Relevance Judgments**
- 52 queries × 50+ candidate documents per query
- 3-point scale: 0=not relevant, 1=partially relevant, 2=highly relevant
- Multiple annotators with inter-rater reliability >0.75
- Domain expert review for technical queries

**Query Categories with Expected Performance**
1. **Factual queries (15)**: High precision expected (P@5 > 0.8)
2. **Analytical queries (15)**: Moderate precision (P@5 > 0.6)  
3. **Technical definitions (12)**: High precision (P@5 > 0.8)
4. **Edge cases (10)**: Lower precision acceptable (P@5 > 0.4)

### Query Dataset Structure
```json
{
  "metadata": {
    "total_queries": 52,
    "categories": ["factual", "analytical", "technical", "edge_case"],
    "creation_date": "2025-08-30",
    "annotators": 3,
    "inter_rater_reliability": 0.82
  },
  "queries": [
    {
      "query_id": "fact_001",
      "category": "factual",
      "query_text": "What is machine learning?",
      "expected_difficulty": "easy",
      "ground_truth": {
        "doc_123": 2,  
        "doc_456": 1,
        "doc_789": 2
      }
    }
  ]
}
```

## Statistical Analysis Framework

### Experimental Design
**Sample Size Calculation**
- 52 queries × 10 runs × 20 configs = 10,400 total evaluations
- Power analysis: >95% power to detect 5% differences in key metrics
- Effect size threshold: Cohen's d > 0.3 (small to medium effect)
- Alpha level: 0.05 with Bonferroni correction for multiple comparisons

**Statistical Tests**
1. **Paired t-tests**: Compare configs on same query set
2. **ANOVA**: Overall configuration effect significance  
3. **Post-hoc tests**: Pairwise config comparisons with correction
4. **Effect size**: Cohen's d for practical significance
5. **Confidence intervals**: 95% CI for all key metrics

### Analysis Procedures
```python
class StatisticalAnalyzer:
    """Statistical analysis for chunking experiments."""
    
    def paired_comparison(self, config_a_results: List[float], 
                         config_b_results: List[float]) -> Dict[str, float]:
        """Paired t-test comparison between configurations."""
        from scipy.stats import ttest_rel
        
        t_stat, p_value = ttest_rel(config_a_results, config_b_results)
        effect_size = self.cohens_d_paired(config_a_results, config_b_results)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'practical_significance': abs(effect_size) > 0.3
        }
    
    def cohens_d_paired(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d for paired samples."""
        diff = [a - b for a, b in zip(group1, group2)]
        return statistics.mean(diff) / statistics.stdev(diff)
    
    def confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for metric."""
        import scipy.stats as stats
        
        mean = statistics.mean(data)
        std_err = statistics.stdev(data) / math.sqrt(len(data))
        margin = stats.t.ppf((1 + confidence) / 2, len(data) - 1) * std_err
        
        return (mean - margin, mean + margin)
```

## Implementation Timeline

### Phase 1: Framework Implementation (2 hours)
- RetrievalQualityEvaluator class
- Ground truth data structure
- Basic metric collection integration

### Phase 2: Advanced Metrics (1.5 hours)  
- Token utilization analysis
- Resource consumption monitoring
- Quality consistency measurements

### Phase 3: Statistical Analysis (1 hour)
- Statistical testing framework
- Effect size calculations
- Confidence interval computation

### Phase 4: Validation and Testing (1 hour)
- Unit tests for all metrics
- Integration testing with experiment runner
- Performance validation

## Expected Outcomes and Benchmarks

### Baseline Expectations
- **Precision@5**: 0.70-0.85 across configurations
- **MRR**: 0.75-0.90 for well-optimized chunking
- **Response time**: 3-8 seconds end-to-end
- **Token efficiency**: 0.08-0.15 for balanced systems

### Optimization Targets
- **10% improvement** in key retrieval metrics over baseline
- **15% reduction** in response time variability  
- **5% better** token utilization efficiency
- **Maintained quality** within 2% of baseline response quality

### Success Criteria
1. ✅ Statistical significance (p < 0.05) for optimal configurations
2. ✅ Practical significance (effect size > 0.3) for key metrics
3. ✅ Quality consistency across configurations (CV < 0.2)
4. ✅ Reproducible results with <5% variance between runs

This comprehensive evaluation framework transforms Experiment 1 from a simple timing study into a rigorous scientific investigation that can definitively identify optimal chunking strategies with statistical confidence and practical significance.