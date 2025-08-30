# Phase 8: Testing Infrastructure Implementation

## Context Loading
**FIRST STEP - MANDATORY**: Read all previous handoff files to understand the complete system:
```bash
for file in docs/handoff/*.json; do echo "=== $file ==="; cat "$file" | jq '.phase, .created_files'; done
```

## Your Mission
Build comprehensive testing infrastructure to measure performance, accuracy, and establish baseline metrics for the RAG system. This phase creates reproducible benchmarks and evaluation frameworks.

## Prerequisites Check
1. Ensure all previous components work: `python main.py stats`
2. Install pytest if needed: `pip install pytest pytest-asyncio pytest-benchmark`
3. Create test data directory: `mkdir -p test_data/corpora`

## Implementation Tasks

### Task 8.1: Performance Benchmark Suite
Create `benchmarks/performance_suite.py`:

```python
# Comprehensive performance benchmarks:
# 1. Token throughput (tokens/second)
# 2. Memory profiling
# 3. Retrieval latency
# 4. End-to-end query latency
# 5. Scaling tests (corpus size impact)
```

Core benchmarks:
```python
import time
import psutil
import statistics
from memory_profiler import profile
from src.rag_pipeline import RAGPipeline

class PerformanceBenchmark:
    def __init__(self):
        self.rag = RAGPipeline()
        self.process = psutil.Process()
        
    def benchmark_token_throughput(self, num_iterations=10):
        # Generate fixed prompt, measure tokens/sec
        # Run multiple times, report mean/std/min/max
        
    def benchmark_memory_usage(self):
        # Track memory during:
        # - Model loading
        # - Document ingestion  
        # - Query processing
        # - Peak usage
        
    def benchmark_retrieval_latency(self, queries, k_values=[1,5,10,20]):
        # Measure retrieval time for different k
        # Test with various query lengths
        
    def benchmark_e2e_latency(self, queries):
        # Complete pipeline timing:
        # - Query processing
        # - Retrieval
        # - Prompt construction
        # - First token
        # - Complete generation
```

Metrics to collect:
- Tokens/second (mean, p50, p95, p99)
- First token latency (milliseconds)
- Time to complete response
- Memory usage (baseline, peak, delta)
- CPU utilization percentage
- Retrieval time vs corpus size

### Task 8.2: Accuracy Evaluation Framework
Create `benchmarks/accuracy_suite.py`:

```python
# RAG accuracy measurements:
# 1. Retrieval relevance scoring
# 2. Answer correctness (if ground truth available)
# 3. Context utilization analysis
# 4. Hallucination detection
```

Evaluation methods:
```python
class AccuracyBenchmark:
    def evaluate_retrieval_relevance(self, queries_with_ground_truth):
        # Measure:
        # - Precision@k
        # - Recall@k  
        # - MRR (Mean Reciprocal Rank)
        # - NDCG (Normalized Discounted Cumulative Gain)
        
    def evaluate_answer_quality(self, qa_pairs):
        # Without ground truth:
        # - Coherence scoring
        # - Context grounding
        # - Response length appropriateness
        
    def detect_hallucinations(self, queries_and_responses):
        # Check if response claims are in retrieved contexts
        # Flag unsupported statements
        
    def analyze_context_usage(self, queries):
        # Measure how much retrieved context is used
        # Identify ignored contexts
```

### Task 8.3: Test Corpus Generation
Create `test_data/generate_test_corpus.py`:

```python
# Generate synthetic test documents:
# 1. Various document types
# 2. Known content for validation
# 3. Different sizes (small/medium/large)
# 4. Edge cases (empty, huge, special chars)
```

Test corpus structure:
```
test_data/corpora/
├── small/      # 10 documents, 1k tokens each
├── medium/     # 100 documents, 5k tokens each  
├── large/      # 1000 documents, mixed sizes
└── edge_cases/ # Problematic documents
```

Document generation:
- Technical articles (Wikipedia-style)
- Q&A formatted content
- Code documentation
- News articles
- Edge cases (Unicode, very long, very short)

### Task 8.4: Benchmark Queries Dataset
Create `test_data/benchmark_queries.json`:

```json
{
  "categories": {
    "factual": [
      {
        "query": "What is machine learning?",
        "expected_topics": ["algorithms", "data", "training"],
        "difficulty": "easy"
      }
    ],
    "analytical": [
      {
        "query": "Compare supervised and unsupervised learning",
        "expected_topics": ["labels", "clustering", "classification"],
        "difficulty": "medium"
      }
    ],
    "edge_cases": [
      {
        "query": "🤖💻🧠",
        "expected_behavior": "handle_gracefully",
        "difficulty": "edge"
      }
    ]
  }
}
```

Query categories:
- Simple factual (single retrieval needed)
- Complex analytical (multiple retrievals)
- Follow-up questions (context-dependent)
- Edge cases (empty, very long, special characters)
- Adversarial (attempting confusion)

### Task 8.5: Automated Test Runner
Create `run_benchmarks.py`:

```python
#!/usr/bin/env python
# Automated benchmark execution:
# 1. Setup test environment
# 2. Run all benchmarks
# 3. Generate reports
# 4. Compare with baselines
```

Features:
```python
class BenchmarkRunner:
    def setup_environment(self):
        # Create test corpus
        # Initialize database
        # Load models
        
    def run_all_benchmarks(self):
        # Performance tests
        # Accuracy tests
        # Stress tests
        # Save results
        
    def generate_report(self, format='markdown'):
        # Create benchmark report
        # Include graphs/tables
        # Compare with previous runs
        
    def check_regressions(self, threshold=0.1):
        # Compare with baseline
        # Flag performance regressions
```

### Task 8.6: Continuous Monitoring
Create `benchmarks/monitoring.py`:

```python
# Runtime monitoring tools:
# 1. Query logging
# 2. Performance tracking
# 3. Error collection
# 4. Usage statistics
```

Monitoring implementation:
- Log all queries and responses
- Track performance metrics per query
- Identify slow queries (>2s)
- Memory leak detection
- Error rate tracking

## Testing Requirements
Create `test_phase_8.py`:
1. Test benchmark suite execution
2. Verify metric collection accuracy
3. Test report generation
4. Validate test corpus generation
5. Check regression detection

## Output Requirements
Create `handoff/phase_8_complete.json`:
```json
{
  "timestamp": "ISO-8601 timestamp",
  "phase": 8,
  "created_files": [
    "benchmarks/performance_suite.py",
    "benchmarks/accuracy_suite.py",
    "benchmarks/monitoring.py",
    "test_data/generate_test_corpus.py",
    "test_data/benchmark_queries.json",
    "run_benchmarks.py",
    "test_phase_8.py"
  ],
  "test_infrastructure": {
    "performance_benchmarks": true,
    "accuracy_evaluation": true,
    "test_corpus_generated": true,
    "automated_runner": true,
    "monitoring": true
  },
  "baseline_metrics": {
    "tokens_per_second": {
      "mean": 0.0,
      "std": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "retrieval_latency_ms": {
      "p50": 0.0,
      "p95": 0.0,
      "p99": 0.0
    },
    "memory_usage_mb": {
      "baseline": 0.0,
      "peak": 0.0
    },
    "e2e_latency_ms": {
      "first_token": 0.0,
      "complete_response": 0.0
    }
  },
  "test_corpora": {
    "small": {"documents": 10, "total_tokens": 10000},
    "medium": {"documents": 100, "total_tokens": 500000},
    "large": {"documents": 1000, "total_tokens": 5000000}
  },
  "benchmark_queries": {
    "total": 50,
    "categories": ["factual", "analytical", "edge_cases"]
  }
}
```

## Benchmark Report Template
Create `benchmark_report_template.md`:
```markdown
# RAG System Benchmark Report
Date: {timestamp}

## Performance Metrics
- Throughput: {tokens_per_second} tokens/sec
- First Token: {first_token_ms}ms  
- Memory Usage: {memory_mb}MB

## Accuracy Metrics
- Retrieval Precision@5: {precision}
- Retrieval Recall@5: {recall}
- Context Utilization: {utilization}%

## Comparison with Baseline
{regression_analysis}

## Recommendations
{performance_suggestions}
```

## Validation Checklist
- [ ] All benchmarks run without errors
- [ ] Metrics are collected and stored
- [ ] Test corpus is generated successfully
- [ ] Reports are generated in markdown
- [ ] Baseline metrics established
- [ ] Regression detection works
- [ ] Monitoring captures live metrics
- [ ] Handoff file created

Remember: These benchmarks establish the "unsafe baseline" for security testing in the next phase.