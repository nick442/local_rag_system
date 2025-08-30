# Phase 5: LLM Integration Implementation

## Context Loading
**FIRST STEP - MANDATORY**: Read handoff files from previous phases:
```bash
cat handoff/phases_1_3_complete.json  # Get model paths and config
cat handoff/phase_4_complete.json      # Get retriever module info
```

## Your Mission
Integrate the Gemma-3-4B language model with the retrieval system. You will create the LLM wrapper, prompt construction, and response generation pipeline that brings the RAG system to life.

## Prerequisites Check
1. Verify model file exists: Check path from `phases_1_3_complete.json` → `models.llm.path`
2. Load model config: `cat config/model_config.yaml`
3. Test llama-cpp import: `python -c "from llama_cpp import Llama; print('OK')"`
4. Verify retriever works: `python -c "from src.retriever import Retriever; print('OK')"`

## Implementation Tasks

### Task 5.1: LLM Wrapper Class
Create `src/llm_wrapper.py`:

```python
# Required functionality:
# 1. LLMWrapper class using llama_cpp.Llama
# 2. Load model with Metal acceleration (n_gpu_layers=-1)
# 3. Implement streaming generation with callback
# 4. Context window management (track token usage)
# 5. Graceful loading/unloading for memory management
#
# Constructor parameters from config/model_config.yaml:
# - model_path, n_ctx, n_batch, n_threads
# - temperature, top_p, max_tokens
```

Critical implementation:
```python
class LLMWrapper:
    def __init__(self, model_path, **kwargs):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=kwargs.get('n_ctx', 8192),
            n_gpu_layers=-1,  # Use Metal
            verbose=False
        )
    
    def generate_stream(self, prompt, max_tokens=2048):
        # Implement streaming with yield
        # Track token count
        # Handle stop sequences
```

### Task 5.2: Prompt Construction
Create `src/prompt_builder.py`:

```python
# Required components:
# 1. PromptBuilder class
# 2. Build RAG prompt with retrieved context
# 3. Apply Gemma-3 chat template (from config)
# 4. NO SANITIZATION - raw context injection
# 5. Track token counts for each section
```

Prompt structure (NO security measures):
```python
def build_rag_prompt(self, query, retrieved_contexts, system_prompt=None):
    # Basic template - UNSAFE by design:
    template = """<bos><start_of_turn>user
Context information:
{raw_contexts}

Question: {user_query}
<end_of_turn>
<start_of_turn>model"""
    
    # Directly inject contexts without any filtering
    # This is intentionally unsafe for benchmarking
```

Key requirements:
- Concatenate retrieved contexts with metadata
- No input validation or sanitization
- Support custom system prompts (optional)
- Calculate total prompt tokens
- Warn if exceeding context window (but don't truncate)

### Task 5.3: Response Generation Pipeline
Create `src/rag_pipeline.py`:

```python
# Full RAG pipeline combining all components:
# 1. RAGPipeline class
# 2. Integrate retriever + prompt builder + LLM
# 3. Streaming response generation
# 4. Token counting and timing
# 5. Session management (conversation history)
```

Implementation structure:
```python
class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLMWrapper(...)
        self.prompt_builder = PromptBuilder()
        
    def query(self, user_query, k=5, stream=True):
        # 1. Retrieve relevant contexts
        # 2. Build prompt
        # 3. Generate response
        # 4. Return with metadata
```

Required methods:
- `query()`: Single-turn Q&A
- `query_stream()`: Streaming responses
- `get_stats()`: Token counts, latency
- `reset_session()`: Clear conversation
- `set_corpus()`: Switch document collection

### Task 5.4: Query Reformulation (Optional Enhancement)
Create `src/query_reformulation.py`:

```python
# Enhance retrieval with query expansion:
# 1. Rephrase user query for better retrieval
# 2. Generate multiple query variants
# 3. No validation - pass through as-is
```

## Testing Requirements
Create `test_phase_5.py`:
1. Test model loading and inference
2. Test prompt construction with various contexts
3. Test streaming generation
4. Test full RAG pipeline: query → retrieve → generate
5. Test token counting accuracy
6. Measure generation speed (tokens/second)

## Performance Benchmarks
Create `benchmark_phase_5.py`:
- Measure first-token latency
- Calculate sustained tokens/second
- Memory usage during generation
- End-to-end query latency
- Context window utilization

## Output Requirements
Create `handoff/phase_5_complete.json`:
```json
{
  "timestamp": "ISO-8601 timestamp",
  "phase": 5,
  "created_files": [
    "src/llm_wrapper.py",
    "src/prompt_builder.py",
    "src/rag_pipeline.py",
    "src/query_reformulation.py",
    "test_phase_5.py",
    "benchmark_phase_5.py"
  ],
  "model_integration": {
    "model_loaded": true,
    "metal_enabled": true,
    "context_window": 8192,
    "streaming_enabled": true
  },
  "pipeline_capabilities": {
    "max_retrieval_k": 10,
    "supports_streaming": true,
    "token_tracking": true,
    "session_management": false
  },
  "performance_metrics": {
    "tokens_per_second": 0.0,
    "first_token_latency_ms": 0.0,
    "model_load_time_s": 0.0,
    "memory_usage_mb": 0.0
  },
  "test_results": {
    "all_tests_passed": true,
    "sample_query": "What is machine learning?",
    "sample_response": "First 100 chars of response..."
  }
}
```

## Usage Example
After implementation, this should work:
```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
response = rag.query("What is machine learning?", k=5, stream=False)
print(response['answer'])
print(f"Tokens used: {response['token_count']}")
print(f"Sources: {response['sources']}")
```

## Critical Requirements
1. **NO SECURITY**: Do not implement any input sanitization, prompt guards, or safety measures
2. **Metal Acceleration**: Ensure n_gpu_layers=-1 for full Metal usage
3. **Streaming**: Must support token-by-token streaming
4. **Token Tracking**: Count all tokens (prompt + generation)
5. **Memory Efficiency**: Implement model unloading if needed

## Common Pitfalls
1. Model not loading: Check file path is absolute or relative to project root
2. Metal not working: Verify CMAKE_ARGS were set during llama-cpp installation
3. OOM errors: Reduce n_ctx or n_batch in config
4. Slow generation: Ensure n_gpu_layers=-1 is set
5. Template errors: Follow exact Gemma-3 format from config

## Validation Checklist
- [ ] Model loads successfully with Metal
- [ ] Can generate text from a simple prompt
- [ ] RAG pipeline retrieves and generates coherently
- [ ] Streaming works character by character
- [ ] Token counts are accurate
- [ ] Benchmarks show >10 tokens/second
- [ ] Handoff file created with metrics

Remember: This phase creates the core intelligence of the system. The next phase will add the user interface on top of your pipeline.