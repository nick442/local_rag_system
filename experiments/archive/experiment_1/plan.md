# Experiment 1: Document Chunking Strategy Optimization Plan

## Overview
Execute the first priority experiment from the research plan to systematically optimize document chunking parameters (chunk_size and chunk_overlap) using the ParametricRAG framework. This will establish optimal chunking strategies for the Mac mini M4 with 16GB RAM configuration.

## Phase 1: System Verification & Preparation (30-45 minutes)

### 1.1 Environment Setup
- Activate conda environment: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`
- Verify ParametricRAG CLI functionality: `python main.py --help` and `python main.py experiment template --list-templates`
- Check system resources: `python main.py stats system`

### 1.2 Corpus Assessment & Preparation
- Check existing collections: `python main.py collection list`
- Verify corpus adequacy (need minimum 1000 documents for meaningful results)
- If insufficient corpus: Use existing test corpus or generate using `test_data/generate_test_corpus.py`
- Create dedicated research collection: `python main.py collection create --id chunking_experiment`

### 1.3 Query Dataset Selection
- Primary dataset: Use `test_data/benchmark_queries.json` (50 comprehensive queries across factual, analytical, and edge cases)
- Fallback: Use built-in DEFAULT_EVALUATION_QUERIES and TECHNICAL_EVALUATION_QUERIES
- Extract query list from structured JSON for CLI usage

## Phase 2: Baseline Establishment (45-60 minutes)

### 2.1 Current Configuration Documentation
- Record current settings: chunk_size=512, chunk_overlap=128, retrieval_k=5
- Run baseline experiment: `python main.py experiment sweep --param temperature --values 0.8 --queries [selected_queries] --output results/baseline_performance.json`

### 2.2 System Performance Validation
- Test basic query functionality: `python main.py query "What is machine learning?" --collection [selected_collection]`
- Verify all components initialize (embedding service, vector database, LLM wrapper)
- Monitor baseline memory usage and response times

## Phase 3: Chunking Optimization Experiment (3-4 hours)

### 3.1 Primary Experiment: Template-Based Approach
- Execute comprehensive chunking optimization: `python main.py experiment template chunk_optimization --output results/chunk_optimization_experiment.json --queries [selected_queries]`
- This tests the pre-defined parameter ranges and provides statistical analysis

### 3.2 Targeted Parameter Sweeps (if needed)
- **Chunk Size Optimization**: Test [128, 256, 512, 768, 1024, 2048] tokens
- **Overlap Optimization**: Test [32, 64, 128, 192, 256] tokens  
- Use individual sweeps: `python main.py experiment sweep --param chunk_size --values 128,256,512,768,1024,2048 --queries [selected_queries] --output results/chunk_size_sweep.json`

### 3.3 Resource Monitoring
- Monitor system memory usage during experiments
- Watch for memory pressure with larger chunk sizes (1024, 2048 tokens)
- Document any performance degradation or failures

## Phase 4: Analysis & Validation (1-2 hours)

### 4.1 Result Analysis
- Analyze experiment outputs using built-in statistical tools
- Compare configurations against baseline performance
- Identify optimal chunk_size and chunk_overlap combinations
- Calculate statistical significance (p < 0.05) and effect sizes (Cohen's d > 0.5)

### 4.2 Success Criteria Validation
- Verify >15% improvement in retrieval precision for best configuration
- Confirm response quality maintained within 5% of baseline
- Ensure all parameter combinations completed within memory constraints
- Document reproducibility (<10% variance between runs)

### 4.3 Configuration Testing
- Test optimal configuration with validation queries
- Measure end-to-end performance improvement
- Document recommended settings for production use

## Expected Outcomes

### Performance Improvements
- Identify optimal chunk_size (likely 256-512 tokens based on literature)
- Determine best chunk_overlap ratio (typically 10-25% of chunk_size)
- Quantify retrieval precision improvements (target: >15%)
- Establish memory usage patterns for different configurations

### Deliverables
- **Baseline Performance Report**: Current system capabilities
- **Optimization Results**: Statistical analysis of all parameter combinations
- **Recommended Configuration**: Evidence-based optimal settings
- **Performance Validation**: End-to-end testing of optimal configuration
- **Resource Usage Profile**: Memory and timing characteristics

### Risk Mitigation
- **Memory Issues**: Monitor usage, reduce batch sizes if needed
- **Experiment Timeouts**: Use smaller query sets for initial testing
- **Statistical Validity**: Ensure minimum 3 runs per configuration
- **Corpus Adequacy**: Verify sufficient document count before starting

## Success Criteria
1. Complete all planned parameter combinations within memory constraints
2. Achieve >15% improvement in retrieval precision
3. Maintain response quality within 5% of baseline  
4. Generate statistically significant results with proper confidence intervals
5. Establish reproducible configuration recommendations for 16GB Mac systems

This plan leverages the existing ParametricRAG infrastructure while ensuring rigorous experimental design and meaningful results for consumer hardware optimization.