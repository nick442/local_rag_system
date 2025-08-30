# Experiment 2: Embedding and LLM Model Comparison Study
## Comprehensive Implementation Plan

### Executive Summary

This experiment systematically evaluates embedding and language model combinations to identify optimal configurations for consumer hardware RAG deployment within 16GB memory constraints. Using the ParametricRAG framework, we will test 3 embedding models × 3 LLM models across multiple performance dimensions to establish evidence-based model selection guidelines.

**Primary Research Question**: Which embedding/LLM combinations provide the best performance-per-resource ratio on Mac mini M4 hardware?

**Key Hypothesis**: Larger embedding models (e5-base-v2, mpnet-base) will improve retrieval precision but create memory pressure that degrades overall system performance. Mid-sized models may achieve optimal balance.

---

## Detailed Experimental Design

### Model Matrix to Evaluate

**Embedding Models** (ordered by resource requirements):
1. **sentence-transformers/all-MiniLM-L6-v2** (Baseline)
   - Dimensions: 384, Size: ~80MB
   - Current system default, established performance baseline
   
2. **sentence-transformers/all-mpnet-base-v2** (Mid-tier)
   - Dimensions: 768, Size: ~400MB
   - Balance of quality and resource usage
   
3. **sentence-transformers/e5-base-v2** (High-quality)
   - Dimensions: 768, Size: ~440MB
   - State-of-the-art performance, highest resource usage

**LLM Models** (ordered by resource requirements):
1. **llama-3.2-3b-instruct-q4_0.gguf** (Efficient)
   - Size: ~2.5GB, Fast inference
   - Lower memory footprint for testing scaling limits
   
2. **gemma-3-4b-it-q4_0.gguf** (Baseline)
   - Size: ~3GB, Current system default
   - Established performance baseline
   
3. **mistral-7b-instruct-q4_0.gguf** (High-capacity)
   - Size: ~5GB, Higher quality potential
   - Tests upper memory limits of 16GB system

**Total Combinations**: 9 model pairs (3×3 matrix)

### Pre-Experiment Validation Phase

**Step 1: System Resource Baseline**
```bash
# Document current system state
python main.py stats system --export results/experiment2_baseline_resources.json

# Test current configuration performance
python main.py experiment sweep \
  --param temperature --values 0.8 \
  --queries test_data/evaluation_queries.json \
  --output results/experiment2_baseline_performance.json
```

**Step 2: Model Availability Check**
```bash
# Verify LLM models exist
ls -la models/ | grep -E "(llama-3.2|gemma-3|mistral-7b)"

# Test embedding model downloads
python -c "
from sentence_transformers import SentenceTransformer
models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'e5-base-v2']
for model in models:
    try:
        SentenceTransformer(f'sentence-transformers/{model}')
        print(f'✓ {model} available')
    except Exception as e:
        print(f'✗ {model} failed: {e}')
"
```

**Step 3: Memory Capacity Testing**
```bash
# Test largest model combination first (safety check)
python main.py config override embedding_model_path "sentence-transformers/e5-base-v2"
python main.py config override llm_model_path "models/mistral-7b-instruct-q4_0.gguf"
python main.py query "Test query for memory validation" --monitor-memory
```

### Core Experiment Execution

**Phase 1: Individual Model Component Testing** (Day 1-2)

*Rationale: Isolate embedding vs LLM contributions before testing combinations*

```bash
# Test embedding models with fixed LLM (baseline Gemma-3-4B)
python main.py experiment sweep \
  --param embedding_model_path \
  --values "sentence-transformers/all-MiniLM-L6-v2,sentence-transformers/all-mpnet-base-v2,sentence-transformers/e5-base-v2" \
  --queries test_data/evaluation_queries.json \
  --output results/experiment2_embedding_comparison.json \
  --monitor-resources

# Test LLM models with fixed embedding (baseline MiniLM-L6-v2)  
python main.py experiment sweep \
  --param llm_model_path \
  --values "models/llama-3.2-3b-instruct-q4_0.gguf,models/gemma-3-4b-it-q4_0.gguf,models/mistral-7b-instruct-q4_0.gguf" \
  --queries test_data/evaluation_queries.json \
  --output results/experiment2_llm_comparison.json \
  --monitor-resources
```

**Phase 2: Full Model Combination Matrix** (Day 3-4)

*Rationale: Comprehensive evaluation of all 9 combinations to identify interaction effects*

```bash
# Run complete model comparison template
python main.py experiment template model_comparison \
  --output results/experiment2_full_matrix.json \
  --queries test_data/evaluation_queries.json \
  --queries test_data/technical_queries.json \
  --monitor-resources \
  --statistical-analysis
```

**Phase 3: Performance Optimization** (Day 5)

*Rationale: Fine-tune best configurations and validate findings*

```bash
# Based on Phase 2 results, test top 3 configurations with extended queries
# (Commands will be determined by Phase 2 analysis)

# Example optimization run for best configuration:
python main.py experiment sweep \
  --param temperature \
  --values 0.6,0.7,0.8,0.9 \
  --model-config "best_embedding_model,best_llm_model" \
  --queries test_data/comprehensive_queries.json \
  --output results/experiment2_optimization.json
```

### Evaluation Framework

**Primary Metrics** (measured for each configuration):

1. **Performance Metrics**:
   - Response time (end-to-end query processing)
   - Tokens per second (generation speed)
   - Memory usage (peak RAM during operation)
   - GPU utilization (Metal acceleration efficiency)

2. **Quality Metrics**:
   - Retrieval precision@5 (relevance of retrieved documents)
   - Response completeness (information coverage)
   - Response accuracy (factual correctness)
   - Response coherence (logical flow and readability)

3. **Resource Efficiency Metrics**:
   - Performance per GB memory used
   - Quality score per processing second
   - Successful query completion rate
   - System stability under load

**Query Test Sets**:

*Set A: General Knowledge* (10 queries)
```
- "What is machine learning?"
- "How does artificial intelligence work?"
- "Explain deep learning algorithms."
- "What are neural networks?"
- "How do large language models work?"
- "What is the difference between AI and machine learning?"
- "How do recommendation systems work?"
- "What are the applications of natural language processing?"
- "Explain computer vision technology."
- "What is reinforcement learning?"
```

*Set B: Technical Implementation* (10 queries)
```
- "How do you implement a neural network from scratch?"
- "What are best practices for machine learning model deployment?"
- "How do you optimize model performance in production?"
- "What are the key considerations for scaling AI systems?"
- "How do you handle data preprocessing for machine learning?"
- "What are the challenges in training large language models?"
- "How do you implement efficient vector search algorithms?"
- "What are the trade-offs between model accuracy and speed?"
- "How do you debug and troubleshoot ML pipelines?"
- "What are the security considerations for AI systems?"
```

*Set C: Complex Analysis* (5 queries)
```
- "Compare the advantages and disadvantages of transformer architectures vs RNNs for sequence modeling tasks."
- "Analyze the computational complexity trade-offs between different attention mechanisms in modern language models."
- "Evaluate the effectiveness of different regularization techniques for preventing overfitting in deep learning models."
- "Discuss the implications of model compression techniques on performance and accuracy for edge deployment."
- "Examine the ethical considerations and bias mitigation strategies in AI system development and deployment."
```

### Statistical Analysis Plan

**Experimental Design**:
- Within-subjects design (same queries tested across all configurations)
- Randomized query presentation order
- Minimum 3 replications per configuration
- Controlled environment (closed applications, consistent system state)

**Statistical Tests**:
1. **ANOVA** for overall model effect significance
2. **Paired t-tests** for pairwise configuration comparison
3. **Effect size calculation** (Cohen's d) for practical significance
4. **Correlation analysis** between resource usage and performance metrics

**Significance Criteria**:
- Statistical significance: p < 0.05
- Practical significance: Effect size > 0.5 OR >15% improvement
- Reliability threshold: Coefficient of variation < 20%

### Resource Management Strategy

**Memory Optimization**:
```bash
# Before each model test, clear system state
sudo purge  # Clear macOS memory pressure
python main.py system restart-components

# Monitor memory during experiments
python main.py stats system --monitor --interval 30 > results/memory_monitoring.log &
```

**Thermal Management**:
- 10-minute cool-down between heavy model loads
- Monitor CPU temperature during extended runs
- Reduce load if thermal throttling detected

**Storage Management**:
- Pre-allocate 5GB for experiment results
- Compress completed result files
- Archive baseline experiments before starting

### Risk Mitigation and Contingencies

**Memory Exhaustion Scenarios**:

*Level 1: Mild Memory Pressure (12-14GB usage)*
- Reduce batch sizes in configuration
- Close background applications
- Continue with current test matrix

*Level 2: Severe Memory Pressure (>14GB usage)*
- Skip largest model combinations (e5-base + Mistral-7B)
- Focus on 6 viable combinations
- Document limitations in results

*Level 3: System Instability (Crashes/Hangs)*
- Revert to sequential testing (one model pair at a time)
- Implement forced garbage collection between tests
- Consider alternative model quantization levels

**Model Loading Failures**:
```bash
# Fallback model verification
python main.py model verify --all-models
python main.py model download --model llama-3.2-3b-instruct-q4_0 --force
```

**Experiment Timeout Handling**:
```bash
# Split large experiments into smaller chunks
python main.py experiment sweep --param embedding_model_path --values "model1" --timeout 3600
python main.py experiment sweep --param embedding_model_path --values "model2" --timeout 3600
```

### Expected Results and Success Criteria

**Performance Hypotheses to Test**:

*H1*: Larger embedding models will improve retrieval quality by 10-25% but increase memory usage by 3-5x

*H2*: Mid-sized LLMs (3-4B parameters) will provide optimal performance/resource ratio compared to smaller (3B) or larger (7B) alternatives

*H3*: Model combinations will show interaction effects where optimal embedding/LLM pairs outperform individual optimized components

**Success Metrics**:
- Complete evaluation of at least 6/9 model combinations within memory constraints
- Identify configuration improving overall performance by >20% vs baseline
- Establish statistical significance for top 3 configurations
- Generate resource utilization profiles for deployment guidance

**Deliverables**:
1. **Performance Matrix**: Complete results for all tested combinations
2. **Resource Profiles**: Memory and compute requirements per configuration
3. **Deployment Guidelines**: Recommended configurations for different use cases
4. **Statistical Report**: Significance tests and confidence intervals

### Implementation Timeline

**Day 1**: Environment preparation and baseline establishment
- System verification and resource documentation
- Model availability validation  
- Baseline performance measurement

**Day 2-3**: Individual component testing
- Embedding model comparison with fixed LLM
- LLM model comparison with fixed embedding
- Initial resource usage analysis

**Day 4-5**: Full combination matrix testing
- Complete 9-combination evaluation (if memory permits)
- Resource monitoring and performance optimization
- Statistical analysis of results

**Day 6**: Results analysis and validation
- Performance ranking and significance testing
- Resource efficiency calculations
- Deployment recommendation generation

**Estimated Total Runtime**: 15-20 hours of active experimentation across 6 days

### Quality Assurance Measures

**Reproducibility Controls**:
- Fixed random seeds for consistent results
- Documented system state before each test
- Version control of all configuration files
- Comprehensive logging of all commands executed

**Validation Steps**:
- Cross-validation of top performers with fresh query sets
- Verification of statistical assumptions (normality, independence)
- Sanity checks on memory usage measurements
- Manual validation of response quality for edge cases

**Documentation Standards**:
- Real-time experiment logging to `results/experiment2_log.txt`
- Automated backup of result files every 2 hours
- Photo documentation of system monitoring displays
- Version tracking of all model files and configurations used

---

## Conclusion

This comprehensive plan for Experiment 2 provides systematic evaluation of embedding/LLM model combinations while respecting the hardware constraints of consumer deployment. The structured approach balances thorough investigation with practical resource management, ensuring reliable results that will inform optimal RAG system configuration for 16GB Mac systems.

The combination of individual component analysis, full matrix evaluation, and statistical rigor will generate actionable insights for both academic understanding and practical deployment scenarios.