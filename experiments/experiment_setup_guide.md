# ParametricRAG Experiment Setup Guide

This guide provides essential context for implementing the experiments outlined in `research_plan.md`. It explains the system architecture, data preparation, and verification steps needed for a complete understanding of the experimental framework.

## System Architecture Overview

### Core Components

**ParametricRAG** is a comprehensive experimental framework for RAG system optimization that consists of:

```
local_rag_system/
├── main.py                     # CLI entry point with experiment commands
├── src/
│   ├── experiment_runner.py    # Core experimental framework and orchestration
│   ├── config_manager.py       # Configuration classes and parameter management
│   ├── experiment_templates.py # Pre-defined experiment templates
│   ├── system_manager.py       # Component lifecycle and resource management
│   ├── rag_pipeline.py         # Main RAG pipeline implementation
│   ├── vector_database.py      # SQLite-based vector storage
│   ├── embedding_service.py    # Embedding model management
│   └── llm_wrapper.py          # Local LLM integration (llama.cpp)
├── config/
│   ├── rag_config.yaml         # Main system configuration
│   ├── model_config.yaml       # Model-specific settings
│   └── app_config.yaml         # Application profiles
├── test_data/                  # Evaluation queries and test datasets
├── results/                    # Experiment outputs and analysis
└── models/                     # Local LLM model files (.gguf)
```

### Key Classes and Their Roles

**ExperimentRunner** (`src/experiment_runner.py`):
- Orchestrates parameter sweeps and A/B testing
- Manages resource allocation and constraint validation
- Provides statistical analysis and result persistence
- Handles experiment database operations

**ExperimentConfig** (`src/config_manager.py`):
- Extends ProfileConfig with 45+ experimental parameters
- Covers document processing, embeddings, retrieval, and generation settings
- Provides parameter validation and resource estimation

**ExperimentTemplate** (`src/config_manager.py`):
- Pre-defined experimental setups for common research patterns
- Includes parameter ranges, evaluation queries, and expected runtimes
- 9 built-in templates: chunk_optimization, model_comparison, retrieval_methods, etc.

**SystemManager** (`src/system_manager.py`):
- Central coordination for RAG system components
- Lazy-loading of components (RAG pipeline, vector database, embedding service)
- Health checking and resource monitoring
- Graceful component lifecycle management

### CLI Interface Structure

The main CLI interface provides experiment-focused commands:

```bash
python main.py experiment --help
# Available subcommands:
#   sweep     - Parameter sweeps across value ranges
#   compare   - A/B testing between configurations  
#   template  - Run pre-defined experiment templates
#   list      - Browse experiment history
```

**Template System**: 
```bash
python main.py experiment template --list-templates
# Shows 9 available templates with descriptions and expected runtimes
python main.py experiment template <template_name>
# Executes pre-configured experimental setup
```

**Parameter Sweeps**:
```bash
python main.py experiment sweep --param <parameter_name> --values <value1,value2,value3>
# Systematic testing across parameter ranges with statistical analysis
```

## Data and Model Preparation

### Current System State

The system is already configured with:
- **LLM Model**: Gemma-3-4B Q4_0 GGUF located at `models/gemma-3-4b-it-q4_0.gguf`
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (downloaded automatically)
- **Vector Database**: SQLite with sqlite-vec extension for efficient similarity search
- **Configuration**: Optimized profiles (fast/balanced/quality) in `config/rag_config.yaml`

### Evaluation Datasets

**Built-in Query Sets** (located in `src/experiment_templates.py`):

1. **DEFAULT_EVALUATION_QUERIES** (10 queries):
   ```python
   [
       "What is machine learning?",
       "How does artificial intelligence work?", 
       "Explain deep learning algorithms.",
       "What are neural networks?",
       "How do large language models work?",
       # ... 5 more general AI questions
   ]
   ```

2. **TECHNICAL_EVALUATION_QUERIES** (10 queries):
   ```python
   [
       "How do you implement a neural network?",
       "What are the best practices for machine learning deployment?",
       "How do you optimize model performance?",
       # ... 7 more technical implementation questions
   ]
   ```

3. **Test Data Files**:
   - `test_data/simple_queries.json` - Basic 5-query test set
   - `test_data/evaluation_queries.json` - Can be created from built-in sets

### Corpus Preparation

**Current Corpus Status**: The system may have existing document collections. Check with:
```bash
python main.py collection list
python main.py collection stats --id <collection_name>
```

**For New Corpus Preparation**:
```bash
# Create new collection for experiments
python main.py collection create --id research_corpus

# Ingest documents (if you have a corpus directory)
python main.py ingest directory corpus/ --collection research_corpus --deduplicate --resume

# Verify corpus is ready
python main.py collection stats --id research_corpus --verbose
```

**Corpus Requirements for Meaningful Results**:
- **Minimum**: 100 documents for basic testing
- **Recommended**: 1,000+ documents for statistical significance
- **Optimal**: 10,000+ documents for comprehensive evaluation

### Model Files and Paths

**Current Model Configuration** (from `config/rag_config.yaml`):
```yaml
model:
  llm_path: "models/gemma-3-4b-it-q4_0.gguf"
  llm_type: "gemma-3"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

**Alternative Models for Experiments** (if available):
- `models/llama-3.2-3b-instruct-q4_0.gguf`
- `models/mistral-7b-instruct-q4_0.gguf`
- Additional embedding models are downloaded automatically by sentence-transformers

## System Verification Checklist

### 1. Environment Activation
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
```

### 2. Basic System Health Check
```bash
# Verify CLI is working
python main.py --help

# Check system resources
python main.py stats system

# Verify experiment interface
python main.py experiment template --list-templates
```

Expected output should show 9 available templates:
```
🧪 Available Experiment Templates:
1. chunk_optimization: Systematic study of chunking strategies
2. model_comparison: Compare different embedding and LLM models
3. retrieval_methods: Test vector, keyword, and hybrid retrieval
4. generation_tuning: Optimize temperature, top_k, and other generation parameters
5. context_length_study: Analyze optimal context window sizes
6. preprocessing_impact: Study document preprocessing effects
7. similarity_metrics: Compare different similarity measures
8. prompt_engineering: Systematic prompt optimization
9. performance_scaling: Analysis of system scaling characteristics
```

### 3. Component Initialization Test
```bash
# Test basic query to verify all components load
python main.py query "What is artificial intelligence?" --collection default

# This should:
# - Initialize embedding service
# - Load vector database
# - Initialize LLM wrapper  
# - Process query and return response
```

### 4. Database and Collection Status
```bash
# Check available collections
python main.py collection list

# Verify vector database is working
python main.py collection stats --id <collection_name>

# Test retrieval functionality
python main.py collection search "machine learning" --collection <collection_name> --k 5
```

### 5. Experiment Framework Test
```bash
# Test minimal parameter sweep
python main.py experiment sweep \
  --param temperature \
  --values 0.7,0.8 \
  --queries test_data/simple_queries.json \
  --output results/verification_test.json

# This should complete without errors and generate results file
```

## Configuration Understanding

### Profile System
The system uses three pre-defined profiles in `config/rag_config.yaml`:

**Fast Profile**: Optimized for speed
```yaml
fast:
  retrieval_k: 3
  max_tokens: 512
  temperature: 0.7
  chunk_size: 256
  chunk_overlap: 64
```

**Balanced Profile**: Default balanced settings
```yaml
balanced:
  retrieval_k: 5
  max_tokens: 1024
  temperature: 0.8
  chunk_size: 512
  chunk_overlap: 128
```

**Quality Profile**: Optimized for response quality
```yaml
quality:
  retrieval_k: 10
  max_tokens: 2048
  temperature: 0.9
  chunk_size: 512
  chunk_overlap: 128
```

### Experimental Parameters

The ExperimentConfig class provides 45+ parameters across four categories:

**Document Processing**:
- `chunk_size`, `chunk_overlap`, `chunk_method`
- `preprocessing_steps`, `duplicate_detection_threshold`

**Embedding/Retrieval**:
- `embedding_model_path`, `retrieval_k`, `similarity_threshold`
- `retrieval_method`, `reranking_enabled`

**LLM Generation**:
- `llm_model_path`, `max_tokens`, `temperature`, `top_p`
- `repetition_penalty`, `system_prompt`

**Corpus/Database**:
- `target_corpus`, `collection_filters`, `quality_threshold`
- `database_backend`, `index_build_strategy`

## Result Analysis and Interpretation

### Experiment Output Structure

**Result Files** (JSON format in `results/` directory):
```json
{
  "experiment_id": "sweep_1756542341",
  "experiment_type": "parameter_sweep",
  "status": "completed",
  "total_runtime": 5.7,
  "results": [
    {
      "run_id": "sweep_1756542341_run_0",
      "query": "What is machine learning?",
      "response": "Generated response text...",
      "metrics": {
        "response_time": 5.55,
        "response_length": 35,
        "num_sources": 5,
        "retrieval_success": 1.0,
        "response_generated": 1.0
      }
    }
  ]
}
```

### Key Metrics Explained

**Performance Metrics**:
- `response_time`: End-to-end query processing (seconds)
- `retrieval_success`: Whether relevant sources were found (0.0-1.0)
- `response_generated`: Whether response was successfully generated (0.0-1.0)

**Quality Metrics**:
- `response_length`: Number of words in generated response
- `num_sources`: Count of retrieved source documents
- `retrieval_precision`: Relevance of retrieved sources (when available)

### Statistical Analysis

The system provides built-in statistical analysis:
- **Significance testing**: Paired t-tests for configuration comparison
- **Confidence intervals**: 95% confidence bounds for performance estimates
- **Effect size calculation**: Cohen's d for practical significance assessment

## Common Troubleshooting Scenarios

### Memory Issues
**Symptoms**: Out of memory errors, system slowdown
**Solutions**:
```bash
# Check memory usage
python main.py stats system --memory

# Reduce batch sizes if needed
python main.py config override batch_size 16

# Use smaller models if available
python main.py config override profiles.balanced.llm.model_path "models/smaller_model.gguf"
```

### Model Loading Failures
**Symptoms**: Model not found errors, loading timeouts
**Solutions**:
```bash
# Verify model files exist
ls -la models/

# Test model loading individually
python main.py query "test" --model-path models/gemma-3-4b-it-q4_0.gguf

# Check embedding model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Experiment Execution Issues
**Symptoms**: Experiments timeout or fail partway through
**Solutions**:
```bash
# Run smaller parameter ranges first
python main.py experiment sweep --param chunk_size --values 256,512 --queries test_data/simple_queries.json

# Monitor system resources during execution
python main.py stats system --monitor &

# Use reduced query sets for testing
python main.py experiment template chunk_optimization --queries test_data/simple_queries.json
```

### Database/Collection Issues
**Symptoms**: No search results, empty collections
**Solutions**:
```bash
# Verify collection exists and has content
python main.py collection list
python main.py collection stats --id <collection_name>

# Rebuild indexes if needed
python main.py collection reindex --collection <collection_name> --operation rebuild

# Test basic search functionality
python main.py collection search "test query" --collection <collection_name> --k 3
```

## Hardware-Specific Considerations

### Mac mini M4 Optimization
- **Unified Memory**: Leverages shared CPU/GPU memory efficiently
- **Metal Acceleration**: GPU acceleration for transformer operations  
- **Thermal Management**: Monitor temperatures during long experiments
- **Memory Pressure**: Keep system memory usage below 14GB for stability

### Performance Expectations
- **Query Processing**: 3-6 seconds per query (including model loading)
- **Experiment Duration**: 2-6 hours for complete template execution
- **Memory Usage**: 8-12GB during active experimentation
- **Storage Requirements**: 2-5GB for result files per experiment

## Integration with Research Plan

This setup guide provides the foundational knowledge needed to execute the experiments outlined in `research_plan.md`:

1. **Experiment 1 (Chunking)**: Uses `chunk_optimization` template with built-in parameter ranges
2. **Experiment 2 (Models)**: Uses `model_comparison` template with available model configurations  
3. **Experiment 3 (Retrieval)**: Uses `retrieval_methods` template with vector/keyword/hybrid comparison

All experiments leverage the pre-built ParametricRAG infrastructure, ensuring reliable execution and comprehensive result analysis.

## Ready for Experiment Execution

With this understanding of the system architecture, data preparation, and verification procedures, the experiments can be executed following the detailed instructions in `research_plan.md`. The combination of this setup guide and the research plan provides complete context for implementing systematic RAG optimization studies on consumer hardware.