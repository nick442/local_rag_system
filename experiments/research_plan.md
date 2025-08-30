# ParametricRAG Research Plan: Top 3 Priority Experiments

## Executive Summary

This research plan outlines three high-impact experiments designed to optimize Retrieval-Augmented Generation (RAG) systems for consumer hardware deployment. Using our fully operational **ParametricRAG experimental interface**, we will systematically investigate chunking strategies, model selection, and retrieval methodologies to identify optimal configurations for resource-constrained environments.

**Research Significance**: These experiments address critical gaps in RAG optimization literature, which predominantly focuses on server-class hardware. Our findings will provide evidence-based guidance for deploying high-performance RAG systems on consumer devices like the Mac mini M4 with 16GB RAM constraints.

**Expected Outcomes**: 
- Quantified performance vs. quality trade-offs for different chunking strategies
- Optimal embedding and LLM model combinations for 16GB memory constraints  
- Hybrid retrieval effectiveness compared to single-method approaches
- Reproducible benchmarks for consumer RAG deployment

## Current Experimental Setup

### Hardware Specifications
- **Platform**: Mac mini M4 with 16GB unified memory
- **GPU**: Apple Silicon with Metal acceleration
- **Storage**: NVMe SSD with 100+ GB available for models and datasets

### Software Infrastructure
- **Framework**: ParametricRAG experimental interface (fully operational)
- **Models**: Gemma-3-4B Q4_0 GGUF (primary LLM), sentence-transformers/all-MiniLM-L6-v2 (embeddings)
- **Vector Database**: SQLite with sqlite-vec extension for efficient similarity search
- **Execution Environment**: Python 3.11 with conda environment (`rag_env`)

### ParametricRAG Capabilities
Our experimental interface provides:
- **45+ experimental parameters** across document processing, embeddings, retrieval, and generation
- **9 pre-defined experiment templates** for common research scenarios
- **Professional CLI interface** with progress tracking and result persistence
- **Statistical analysis** integration for significance testing and confidence intervals
- **Database persistence** with comprehensive experiment tracking

### Available Experiment Templates
1. **chunk_optimization**: Systematic chunking strategy analysis
2. **model_comparison**: Embedding and LLM model comparison
3. **retrieval_methods**: Vector, keyword, and hybrid search analysis
4. **generation_tuning**: LLM parameter optimization
5. Additional templates available for extended research

## Experiment 1: Document Chunking Strategy Optimization

### Research Objective
**Primary Question**: How do different document chunking strategies affect retrieval precision and answer quality on resource-constrained hardware?

**Hypothesis**: Smaller chunk sizes (256-512 tokens) will improve retrieval precision due to more focused semantic content, but may reduce generation quality due to limited context per chunk. Optimal overlap ratios will balance information completeness with indexing efficiency.

### Research Background
Prior research by Mattambrogi (2024) demonstrated dramatic improvements in Mean Reciprocal Rank (MRR) from 0.24 to 0.84 when optimizing chunk sizes, but these studies used larger models or cloud infrastructure. Our experiment will establish optimal chunking for 4B models on consumer hardware.

### Implementation Details

**Template**: `chunk_optimization` (pre-defined in ParametricRAG)

**Parameters to Test**:
```bash
chunk_size: [128, 256, 512, 1024, 2048] tokens
chunk_overlap: [32, 64, 128, 256] tokens  
chunking_strategy: ["token-based", "sentence-based", "paragraph-based"]
```

**CLI Execution**:
```bash
# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env

# Run comprehensive chunking experiment
python main.py experiment template chunk_optimization \
  --output results/chunk_optimization_experiment.json \
  --queries test_data/evaluation_queries.json
```

**Alternative Parameter Sweeps**:
```bash
# Focus on chunk size optimization
python main.py experiment sweep \
  --param chunk_size \
  --values 128,256,512,768,1024 \
  --queries test_data/evaluation_queries.json \
  --output results/chunk_size_sweep.json

# Focus on overlap optimization  
python main.py experiment sweep \
  --param chunk_overlap \
  --values 32,64,128,192,256 \
  --queries test_data/evaluation_queries.json \
  --output results/chunk_overlap_sweep.json
```

### Requirements for Meaningful Results

**1. Corpus Preparation**:
```bash
# Ensure adequate test corpus (minimum 1000 documents)
python main.py collection create --id chunking_test
python main.py ingest directory corpus/ --collection chunking_test --deduplicate
python main.py collection stats --id chunking_test  # Verify corpus size
```

**2. Baseline Establishment**:
- Run current configuration first to establish performance baseline
- Document current chunk_size=512, chunk_overlap=128 performance

**3. Evaluation Queries**:
- Use built-in DEFAULT_EVALUATION_QUERIES (10 general AI questions)
- Supplement with domain-specific queries if available
- Ensure queries span different complexity levels (factual, analytical, synthetic)

**4. Statistical Significance**:
- Minimum 3 runs per configuration for statistical validity  
- Track confidence intervals and standard deviations
- Use paired t-tests for configuration comparison

### Evaluation Metrics
- **Retrieval Precision@K**: Relevance of retrieved chunks
- **Retrieval Recall@K**: Coverage of relevant information
- **Response Quality**: Automated scoring using RAGAS framework
- **Response Time**: End-to-end query processing latency
- **Memory Usage**: Peak RAM consumption during processing
- **Index Size**: Storage requirements for different chunk configurations

### Expected Outcomes
- **Runtime**: ~3 hours for complete template execution
- **Key Insights**: Optimal chunk size for 4B models (likely 256-512 tokens)
- **Trade-offs**: Quantified precision vs. context trade-offs
- **Recommendations**: Evidence-based chunking guidelines for 16GB systems

### Success Criteria
- Identify chunk configuration improving retrieval precision by >15%
- Maintain response quality within 5% of baseline
- Complete all parameter combinations within memory constraints
- Generate reproducible results with <10% variance between runs

---

## Experiment 2: Embedding and LLM Model Comparison

### Research Objective
**Primary Question**: Which combination of embedding and language models provides optimal performance for consumer hardware RAG deployment within 16GB memory constraints?

**Hypothesis**: Larger embedding models (e5-base-v2, mpnet-base) will improve retrieval quality but increase memory pressure. Smaller LLMs (Gemma-3-4B) with retrieval may outperform larger models (Mistral-7B) without retrieval on the same hardware.

### Research Background
Limited empirical data exists on model performance trade-offs for consumer hardware RAG deployment. Most benchmarks assume unlimited memory or cloud deployment. Our experiment will quantify performance vs. resource consumption for practical model selection.

### Implementation Details

**Template**: `model_comparison` (pre-defined in ParametricRAG)

**Embedding Models to Test**:
```bash
embedding_model_path: [
  "sentence-transformers/all-MiniLM-L6-v2",      # 384-dim, ~80MB (baseline)
  "sentence-transformers/all-mpnet-base-v2",     # 768-dim, ~400MB  
  "sentence-transformers/e5-base-v2"             # 768-dim, ~440MB
]
```

**LLM Models to Test**:
```bash
llm_model_path: [
  "models/gemma-3-4b-it-q4_0.gguf",             # ~3GB (baseline)
  "models/llama-3.2-3b-instruct-q4_0.gguf",     # ~2.5GB
  "models/mistral-7b-instruct-q4_0.gguf"        # ~5GB (if available)
]
```

**CLI Execution**:
```bash
# Run comprehensive model comparison
python main.py experiment template model_comparison \
  --output results/model_comparison_experiment.json \
  --queries test_data/technical_queries.json
```

**Individual Model Testing**:
```bash
# Test embedding model impact
python main.py experiment sweep \
  --param embedding_model_path \
  --values "sentence-transformers/all-MiniLM-L6-v2,sentence-transformers/all-mpnet-base-v2" \
  --queries test_data/evaluation_queries.json \
  --output results/embedding_model_comparison.json

# Test LLM model impact  
python main.py experiment sweep \
  --param llm_model_path \
  --values "models/gemma-3-4b-it-q4_0.gguf,models/llama-3.2-3b-instruct-q4_0.gguf" \
  --queries test_data/evaluation_queries.json \
  --output results/llm_model_comparison.json
```

### Requirements for Meaningful Results

**1. Model Availability**:
```bash
# Verify all models are available
ls -la models/  # Check LLM models present
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" # Test embedding models
```

**2. Memory Monitoring**:
```bash
# Monitor system resources during experiments
python main.py stats system --export results/system_monitoring.json
```

**3. Controlled Environment**:
- Close unnecessary applications to minimize memory competition
- Use consistent temperature/generation settings across LLM tests
- Ensure identical corpus and chunk configurations

**4. Baseline Measurements**:
- Establish current model performance (Gemma-3-4B + MiniLM-L6-v2)
- Document memory usage patterns for each model combination

### Evaluation Metrics
- **Accuracy**: Response correctness on factual queries
- **Speed**: Tokens per second generation rate
- **Memory Usage**: Peak RAM consumption per model
- **Response Quality**: Automated quality scoring
- **Retrieval Performance**: Precision/recall for different embeddings
- **Resource Efficiency**: Performance per GB of memory used

### Expected Outcomes
- **Runtime**: ~6 hours for complete comparison matrix
- **Key Insights**: Optimal model combinations for different use cases
- **Trade-offs**: Performance vs. memory consumption curves
- **Recommendations**: Model selection guidelines for 16GB constraints

### Success Criteria
- Test all embedding/LLM combinations within memory limits
- Identify configurations improving accuracy by >10% 
- Quantify memory vs. performance trade-offs
- Generate deployment recommendations for different hardware tiers

---

## Experiment 3: Retrieval Method Analysis (Vector vs. Keyword vs. Hybrid)

### Research Objective
**Primary Question**: How do vector, keyword, and hybrid retrieval methods compare in terms of accuracy, speed, and query type sensitivity on consumer hardware?

**Hypothesis**: Hybrid retrieval combining dense vectors with sparse keyword matching will outperform individual methods, particularly for queries containing specific terminology or proper nouns. Performance gains will justify computational overhead on consumer hardware.

### Research Background
Amazon research (2024) demonstrated 12-20% NDCG improvements using hybrid dense+sparse retrieval, but studies focused on cloud-scale deployment. Our experiment will validate hybrid retrieval effectiveness in resource-constrained environments and identify optimal alpha parameters for combining methods.

### Implementation Details

**Template**: `retrieval_methods` (pre-defined in ParametricRAG)

**Parameters to Test**:
```bash
retrieval_method: ["vector", "keyword", "hybrid"]
retrieval_k: [3, 5, 7, 10, 15, 20]
similarity_threshold: [0.0, 0.2, 0.4, 0.6, 0.8] (for hybrid α weighting)
```

**CLI Execution**:
```bash
# Run comprehensive retrieval methods analysis
python main.py experiment template retrieval_methods \
  --output results/retrieval_methods_experiment.json \
  --queries test_data/mixed_queries.json
```

**Focused Method Comparison**:
```bash
# Compare retrieval methods directly
python main.py experiment sweep \
  --param retrieval_method \
  --values "vector,keyword,hybrid" \
  --queries test_data/evaluation_queries.json \
  --output results/method_comparison.json

# Optimize hybrid alpha parameter
python main.py experiment sweep \
  --param similarity_threshold \
  --values "0.3,0.5,0.7,0.9" \
  --queries test_data/evaluation_queries.json \
  --output results/hybrid_alpha_optimization.json
```

### Requirements for Meaningful Results

**1. Corpus Indexing**:
```bash
# Ensure both vector and keyword indexes are built
python main.py collection reindex --collection test_corpus --operation rebuild
python main.py collection stats --collection test_corpus --verbose  # Verify both indexes
```

**2. Query Diversity**:
- Include factual queries (benefit from vector search)
- Include term-specific queries (benefit from keyword search) 
- Include mixed-complexity queries (benefit from hybrid)
- Test across DEFAULT_EVALUATION_QUERIES + TECHNICAL_EVALUATION_QUERIES

**3. Performance Baseline**:
```bash
# Establish single-method baselines first
python main.py query "What is machine learning?" --method vector --collection test_corpus
python main.py query "What is machine learning?" --method keyword --collection test_corpus  
python main.py query "What is machine learning?" --method hybrid --collection test_corpus
```

**4. Statistical Design**:
- Test each method with identical query sets
- Use paired statistical tests for method comparison
- Control for query complexity and type

### Evaluation Metrics
- **Retrieval Precision@K**: Relevance of top-K results
- **Retrieval Recall@K**: Coverage of relevant information
- **Retrieval Speed**: Query processing latency
- **Response Relevance**: End-to-end answer quality
- **Method Win Rate**: Percentage of queries where method performs best
- **NDCG@10**: Normalized Discounted Cumulative Gain

### Expected Outcomes
- **Runtime**: ~2.5 hours for complete analysis
- **Key Insights**: Query-type specific method advantages
- **Optimal Configuration**: Best hybrid alpha parameters
- **Performance Profiles**: Speed vs. quality trade-offs per method

### Success Criteria
- Demonstrate statistical significance in method comparison
- Identify hybrid configurations outperforming single methods by >10%
- Establish query-type routing guidelines 
- Quantify computational overhead for hybrid approach

---

## Execution Timeline and Resource Requirements

### Phase 1: Environment Preparation (Day 1)
1. **System Setup Verification**
   ```bash
   source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
   python main.py experiment template --list-templates  # Verify interface
   python main.py stats system  # Check resource availability
   ```

2. **Corpus Preparation**
   ```bash
   python main.py collection create --id research_corpus
   python main.py ingest directory corpus/ --collection research_corpus --deduplicate
   python main.py collection stats --id research_corpus
   ```

3. **Baseline Establishment**
   ```bash
   python main.py experiment sweep \
     --param temperature --values 0.8 \
     --queries test_data/evaluation_queries.json \
     --output results/baseline_performance.json
   ```

### Phase 2: Chunk Optimization (Days 2-3)
- **Day 2**: Run chunk_optimization template experiment
- **Day 3**: Analyze results, run focused parameter sweeps if needed
- **Expected Runtime**: 3-4 hours active experimentation

### Phase 3: Model Comparison (Days 4-5)  
- **Day 4**: Run model_comparison template experiment
- **Day 5**: Analyze memory usage, run performance validation
- **Expected Runtime**: 6-8 hours active experimentation

### Phase 4: Retrieval Methods (Days 6-7)
- **Day 6**: Run retrieval_methods template experiment
- **Day 7**: Analyze method performance, optimize hybrid parameters
- **Expected Runtime**: 2.5-3 hours active experimentation

### Phase 5: Analysis and Documentation (Days 8-10)
- Comprehensive result analysis using built-in statistical tools
- Performance visualization and comparison
- Research report generation with findings and recommendations

## Statistical Analysis Framework

### Experimental Design Principles
1. **Controlled Variables**: Maintain consistent environment across experiments
2. **Randomization**: Random query ordering to prevent bias
3. **Replication**: Minimum 3 runs per configuration for statistical validity
4. **Paired Comparison**: Use same queries across different configurations

### Statistical Tests
- **Paired t-tests** for configuration comparison
- **ANOVA** for multi-group analysis  
- **Effect size calculation** (Cohen's d) for practical significance
- **Confidence intervals** for performance estimates

### Significance Thresholds
- **Statistical Significance**: p < 0.05
- **Practical Significance**: Effect size > 0.5 OR improvement > 10%
- **Minimum Sample Size**: 30 queries per configuration

## Success Metrics and Deliverables

### Research Output
1. **Quantitative Results**: Performance metrics for all experimental configurations
2. **Statistical Analysis**: Significance tests and confidence intervals
3. **Optimization Guidelines**: Evidence-based recommendations for consumer hardware
4. **Reproducible Benchmarks**: Standardized evaluation protocols

### Expected Contributions
1. **Academic**: First comprehensive study of RAG optimization on consumer hardware
2. **Practical**: Deployment guidelines for 16GB Mac systems
3. **Technical**: Validation of ParametricRAG experimental framework
4. **Community**: Open-source benchmarks and evaluation datasets

### Final Deliverables
- **Research Report**: Comprehensive findings and analysis
- **Performance Data**: All experimental results and statistics  
- **Configuration Recommendations**: Optimal settings for different use cases
- **Reproducibility Package**: Complete experimental setup and data

---

## Troubleshooting and Contingency Plans

### Common Issues and Solutions

**Memory Constraints**:
```bash
# Monitor memory usage during experiments
python main.py stats system --monitor
# Reduce batch sizes if needed
python main.py config override batch_size 16
```

**Model Loading Failures**:
```bash
# Verify model availability
ls -la models/
# Test model loading individually
python main.py query "test" --model-path models/target_model.gguf
```

**Experiment Timeouts**:
```bash
# Run smaller parameter sweeps
python main.py experiment sweep --param chunk_size --values 256,512 --queries test_data/small_query_set.json
```

**Storage Space Issues**:
```bash
# Clean up old experiment results
rm -rf results/old_experiments/
# Monitor disk usage
df -h
```

### Alternative Approaches
- **Reduced Parameter Space**: Focus on most promising parameter ranges
- **Sequential Execution**: Run experiments individually if resource competition occurs
- **Cloud Backup**: Upload large result files to prevent local storage issues

## Conclusion

This research plan provides a comprehensive framework for systematic RAG optimization using the ParametricRAG experimental interface. The three priority experiments will generate actionable insights for consumer hardware deployment while establishing reproducible benchmarks for the research community.

**Key Advantages**:
- Leverages fully operational experimental infrastructure
- Addresses critical gaps in consumer RAG research  
- Provides quantitative optimization guidelines
- Establishes reproducible evaluation methodology

**Expected Impact**:
- Academic contribution to resource-constrained AI research
- Practical deployment guidance for consumer hardware
- Validation of local RAG system viability
- Foundation for extended research in specialized applications

The combination of systematic experimental design, statistical rigor, and practical focus positions this research to make significant contributions to both academic understanding and practical deployment of RAG systems on consumer hardware.