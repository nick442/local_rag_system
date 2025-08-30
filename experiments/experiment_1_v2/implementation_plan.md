# Implementation Plan: Fixing Experiment 1 Architecture Issues

## Overview
This document provides detailed technical implementation steps to fix the critical issues identified in the Experiment 1 review. These fixes will enable valid document chunking experiments by properly implementing chunking parameter materialization, collection isolation, and comprehensive metrics collection.

## Critical Issues Identified
1. **No applied re-chunking per configuration** - ExperimentRunner ignores chunk_size/chunk_overlap parameters
2. **Missing collection isolation** - All configs query the same global data
3. **Incomplete parameter provenance** - Results lack run-time config tracking  
4. **Insufficient metrics** - Only coarse timing, no retrieval quality or token counts
5. **Undersized query set** - Only 3 queries used, statistically invalid
6. **Questionable claims** - 75% overlap recommendation contradicts best practices

## Implementation Phase 1: Core Infrastructure Fixes

### 1.1 Fix ExperimentRunner._create_rag_pipeline()
**File**: `src/experiment_runner.py:501-524`

**Current Problem**: Only overrides `retrieval_k`, `max_tokens`, `temperature`; ignores chunking params.

**Fix**:
```python
def _create_rag_pipeline(self, config: ExperimentConfig) -> RAGPipeline:
    """Create RAG pipeline with experimental configuration."""
    if not self.system_manager:
        from .system_manager import SystemManager
        self.system_manager = SystemManager()
        self.system_manager.initialize_components()
    
    # Create RAG pipeline with paths
    rag_pipeline = RAGPipeline(
        db_path=self.system_manager.config.db_path,
        embedding_model_path=self.system_manager.config.embedding_model_path,
        llm_model_path=self.system_manager.config.llm_model_path
    )
    
    # Override LLM configuration parameters
    if hasattr(config, 'retrieval_k'):
        rag_pipeline.config['retrieval']['default_k'] = config.retrieval_k
    if hasattr(config, 'max_tokens'):
        rag_pipeline.config['llm_params']['max_tokens'] = config.max_tokens
    if hasattr(config, 'temperature'):
        rag_pipeline.config['llm_params']['temperature'] = config.temperature
    
    # NEW: Handle chunking parameters by creating per-config collection
    if hasattr(config, 'chunk_size') or hasattr(config, 'chunk_overlap'):
        collection_id = self._ensure_chunked_collection(config)
        rag_pipeline.set_corpus(collection_id)
    
    return rag_pipeline

def _ensure_chunked_collection(self, config: ExperimentConfig) -> str:
    """Ensure a collection exists with the specified chunking parameters."""
    # Generate unique collection ID for this config
    chunk_size = getattr(config, 'chunk_size', 512)
    chunk_overlap = getattr(config, 'chunk_overlap', 128)
    collection_id = f"exp_cs{chunk_size}_co{chunk_overlap}"
    
    # Check if collection already exists with correct parameters
    existing_collections = self.db.list_collections()
    if collection_id in [c['collection_id'] for c in existing_collections]:
        self.logger.info(f"Reusing existing collection: {collection_id}")
        return collection_id
    
    # Create new collection with proper chunking
    self.logger.info(f"Creating chunked collection: {collection_id} (size={chunk_size}, overlap={chunk_overlap})")
    
    # Use ReindexTool to create properly chunked collection
    from .reindex import ReindexTool
    reindex_tool = ReindexTool(self.system_manager.config.db_path)
    
    # First create base collection from source documents
    source_collection = "realistic_full_production"  # or configurable
    
    # Copy documents to new collection and rechunk
    stats = reindex_tool.rechunk_documents(
        collection_id=collection_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reembed=True,
        backup=False  # Don't backup during experiments
    )
    
    if not stats.success:
        raise RuntimeError(f"Failed to create chunked collection {collection_id}: {stats.details.get('error')}")
    
    self.logger.info(f"Created collection {collection_id}: {stats.documents_processed} docs, {stats.chunks_processed} chunks")
    return collection_id
```

### 1.2 Fix RAGPipeline.query() Collection Threading
**File**: `src/rag_pipeline.py:141-296`

**Current Problem**: collection_id parameter is accepted but not passed to retriever.

**Fix** (line ~180):
```python
# Current code:
contexts = self.retriever.retrieve(
    user_query, 
    k=k, 
    method=retrieval_method
)

# Fixed code:
contexts = self.retriever.retrieve(
    user_query, 
    k=k, 
    method=retrieval_method,
    collection_id=collection_id  # NEW: Thread collection_id
)
```

### 1.3 Fix Retriever.retrieve() Collection Parameter
**File**: `src/retriever.py:63-83`

**Current Problem**: retrieve() method doesn't accept or use collection_id.

**Fix**:
```python
def retrieve(self, query: str, k: int = 5, 
            method: str = "vector", collection_id: Optional[str] = None) -> List[RetrievalResult]:
    """
    Retrieve relevant chunks for a query.
    
    Args:
        query: Search query string
        k: Number of results to return
        method: Retrieval method ("vector", "keyword", "hybrid")
        collection_id: Optional collection filter
        
    Returns:
        List of RetrievalResult objects ordered by relevance
    """
    if method == "vector":
        return self._vector_retrieve(query, k, collection_id)
    elif method == "keyword":
        return self._keyword_retrieve(query, k, collection_id)
    elif method == "hybrid":
        return self._hybrid_retrieve(query, k, collection_id)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

# Update all internal methods:
def _vector_retrieve(self, query: str, k: int, collection_id: Optional[str] = None) -> List[RetrievalResult]:
    # Pass collection_id to vector database
    return self.vector_db.search_similar(query, k=k, collection_id=collection_id)

def _keyword_retrieve(self, query: str, k: int, collection_id: Optional[str] = None) -> List[RetrievalResult]:
    # Pass collection_id to keyword search
    return self.vector_db.search_keyword(query, k=k, collection_id=collection_id)

def _hybrid_retrieve(self, query: str, k: int, collection_id: Optional[str] = None) -> List[RetrievalResult]:
    # Pass collection_id to hybrid search
    return self.vector_db.search_hybrid(query, k=k, collection_id=collection_id)
```

### 1.4 Fix Experiment Result Export with Full Config
**File**: `main.py` (_save_experiment_results function)

**Current Problem**: Results don't include per-run config parameters.

**Fix**:
```python
def _save_experiment_results(results: List[ExperimentResult], output_path: str):
    """Save experiment results with full configuration provenance."""
    export_data = []
    
    for result in results:
        # Include full configuration in each result
        result_dict = {
            'config': {
                'chunk_size': getattr(result.config, 'chunk_size', None),
                'chunk_overlap': getattr(result.config, 'chunk_overlap', None),
                'retrieval_k': getattr(result.config, 'retrieval_k', None),
                'temperature': getattr(result.config, 'temperature', None),
                'max_tokens': getattr(result.config, 'max_tokens', None),
                'collection_id': getattr(result.config, 'collection_id', None),
                'profile': getattr(result.config, 'profile', None)
            },
            'query': result.query,
            'response': result.response,
            'response_length': len(result.response.split()),
            'num_sources': result.num_sources,
            'created_at': result.created_at.isoformat(),
            'completed_at': result.completed_at.isoformat(),
            'total_runtime': result.total_runtime,
            'metadata': result.metadata,
            # NEW: Enhanced metrics
            'retrieval_time': result.metadata.get('retrieval_time'),
            'generation_time': result.metadata.get('generation_time'), 
            'prompt_tokens': result.metadata.get('prompt_tokens'),
            'output_tokens': result.metadata.get('output_tokens'),
            'total_tokens': result.metadata.get('total_tokens'),
            'contexts_count': result.metadata.get('contexts_count'),
            'tokens_per_second': result.metadata.get('tokens_per_second')
        }
        export_data.append(result_dict)
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
```

## Implementation Phase 2: Enhanced Experiment Infrastructure

### 2.1 Create Enhanced Query Dataset
**File**: `test_data/enhanced_evaluation_queries.json`

Create comprehensive query set with 50+ queries across categories:
- Factual questions (15 queries)
- Analytical questions (15 queries)  
- Technical definitions (10 queries)
- Edge cases (ambiguous, multi-part) (10 queries)

### 2.2 Add Retrieval Quality Metrics
**File**: `src/evaluation_metrics.py` (NEW)

```python
class RetrievalQualityEvaluator:
    """Evaluate retrieval quality using standard IR metrics."""
    
    def calculate_precision_at_k(self, retrieved_docs, relevant_docs, k):
        """Calculate Precision@K"""
        
    def calculate_recall_at_k(self, retrieved_docs, relevant_docs, k):
        """Calculate Recall@K"""
        
    def calculate_mrr(self, retrieved_docs_list, relevant_docs_list):
        """Calculate Mean Reciprocal Rank"""
        
    def calculate_ndcg_at_k(self, retrieved_docs, relevance_scores, k):
        """Calculate Normalized Discounted Cumulative Gain@K"""
```

### 2.3 Add Collection Management Utilities
**File**: `src/experiment_utils.py` (NEW)

```python
class ExperimentCollectionManager:
    """Manage collections for chunking experiments."""
    
    def create_experiment_collection(self, base_collection: str, chunk_size: int, chunk_overlap: int) -> str:
        """Create collection with specific chunking parameters."""
        
    def cleanup_experiment_collections(self, pattern: str = "exp_*"):
        """Clean up experimental collections."""
        
    def validate_collection_parameters(self, collection_id: str) -> Dict[str, Any]:
        """Validate that collection has expected chunking parameters."""
```

## Implementation Phase 3: Validation and Testing

### 3.1 Unit Tests for Infrastructure Fixes
**File**: `tests/test_experiment_fixes.py` (NEW)

Test cases:
- Collection ID threading through pipeline
- Per-config collection creation
- Enhanced result export format
- Chunking parameter materialization

### 3.2 Integration Tests
**File**: `tests/test_experiment_v2_integration.py` (NEW)

End-to-end tests:
- Full experiment run with chunking
- Collection isolation validation
- Metrics collection accuracy

### 3.3 Performance Validation
**File**: `tests/test_chunking_performance.py` (NEW)

Validation tests:
- Memory usage with different chunk sizes
- Processing time scaling
- Index size impact

## Success Criteria
1. ✅ ExperimentRunner properly materializes chunking parameters
2. ✅ Collection isolation works end-to-end
3. ✅ Results include full configuration provenance
4. ✅ Enhanced metrics are collected accurately
5. ✅ Query set expanded to 50+ diverse questions
6. ✅ Overlap constraints follow best practices (10-25% of chunk_size)

## Implementation Timeline
- **Phase 1 (Core Fixes)**: 2-3 hours
- **Phase 2 (Enhanced Infrastructure)**: 3-4 hours  
- **Phase 3 (Validation)**: 1-2 hours
- **Total Estimated Time**: 6-9 hours

This implementation plan addresses all critical issues identified in the experiment review and will enable valid, reproducible chunking experiments with proper scientific methodology.