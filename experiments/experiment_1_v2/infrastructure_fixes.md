# Infrastructure Fixes: Code Changes Required for Valid Chunking Experiments

## Overview
This document details the exact code changes required to fix the architectural issues that invalidated Experiment 1. Each section includes the specific files, line numbers, and code modifications needed to enable valid chunking experiments.

## Fix 1: ExperimentRunner Chunking Parameter Handling

### File: `src/experiment_runner.py`
### Issue: Lines 501-524 (_create_rag_pipeline) ignore chunk_size and chunk_overlap
### Impact: HIGH - Core functionality broken

#### Current Problematic Code:
```python
def _create_rag_pipeline(self, config: ExperimentConfig) -> RAGPipeline:
    """Create RAG pipeline with experimental configuration."""
    # ... initialization code ...
    
    # Override configuration parameters
    if hasattr(config, 'retrieval_k'):
        rag_pipeline.config['retrieval']['default_k'] = config.retrieval_k
    if hasattr(config, 'max_tokens'):
        rag_pipeline.config['llm_params']['max_tokens'] = config.max_tokens
    if hasattr(config, 'temperature'):
        rag_pipeline.config['llm_params']['temperature'] = config.temperature
    
    # MISSING: chunk_size and chunk_overlap handling
    return rag_pipeline
```

#### Required Fix:
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
    collection_id = self._ensure_chunked_collection(config)
    rag_pipeline.set_corpus(collection_id)
    
    return rag_pipeline
```

#### Additional Method Required:
```python
def _ensure_chunked_collection(self, config: ExperimentConfig) -> str:
    """Ensure a collection exists with the specified chunking parameters."""
    # Generate unique collection ID for this config
    chunk_size = getattr(config, 'chunk_size', 512)
    chunk_overlap = getattr(config, 'chunk_overlap', 128)
    collection_id = f"exp_cs{chunk_size}_co{chunk_overlap}"
    
    # Check if collection already exists
    try:
        collections = self.db.list_collections()
        if any(c['collection_id'] == collection_id for c in collections):
            self.logger.info(f"Reusing existing collection: {collection_id}")
            return collection_id
    except Exception as e:
        self.logger.warning(f"Could not check existing collections: {e}")
    
    # Create new collection with proper chunking
    self.logger.info(f"Creating chunked collection: {collection_id} (size={chunk_size}, overlap={chunk_overlap})")
    
    # Use ReindexTool to create properly chunked collection
    from .reindex import ReindexTool
    reindex_tool = ReindexTool(self.system_manager.config.db_path)
    
    # Source collection (configurable, default to production)
    source_collection = "realistic_full_production"
    
    # Copy and rechunk documents
    try:
        stats = reindex_tool.rechunk_documents(
            collection_id=collection_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reembed=True,
            backup=False
        )
        
        if not stats.success:
            raise RuntimeError(f"Rechunking failed: {stats.details.get('error', 'Unknown error')}")
            
        self.logger.info(f"Created collection {collection_id}: {stats.documents_processed} docs, {stats.chunks_processed} chunks")
        return collection_id
        
    except Exception as e:
        self.logger.error(f"Failed to create chunked collection {collection_id}: {e}")
        # Fallback to default collection
        return "default"
```

#### Import Addition Required:
Add to imports section:
```python
from typing import Optional  # if not already present
```

## Fix 2: RAGPipeline Collection ID Threading

### File: `src/rag_pipeline.py`  
### Issue: Line ~180 doesn't pass collection_id to retriever
### Impact: HIGH - Collection isolation broken

#### Current Code (around line 180):
```python
# Note: Retriever currently does not accept collection_id; future work can thread it down.
contexts = self.retriever.retrieve(
    user_query, 
    k=k, 
    method=retrieval_method
)
```

#### Required Fix:
```python
# Pass collection_id to retriever for proper isolation
contexts = self.retriever.retrieve(
    user_query, 
    k=k, 
    method=retrieval_method,
    collection_id=collection_id
)
```

## Fix 3: Retriever Collection Parameter Support

### File: `src/retriever.py`
### Issue: Lines 63-83 (retrieve method) doesn't accept collection_id
### Impact: HIGH - Collection filtering impossible

#### Current Code:
```python
def retrieve(self, query: str, k: int = 5, 
            method: str = "vector") -> List[RetrievalResult]:
    """
    Retrieve relevant chunks for a query.
    
    Args:
        query: Search query string
        k: Number of results to return
        method: Retrieval method ("vector", "keyword", "hybrid")
        
    Returns:
        List of RetrievalResult objects ordered by relevance
    """
    if method == "vector":
        return self._vector_retrieve(query, k)
    elif method == "keyword":
        return self._keyword_retrieve(query, k)
    elif method == "hybrid":
        return self._hybrid_retrieve(query, k)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
```

#### Required Fix:
```python
def retrieve(self, query: str, k: int = 5, 
            method: str = "vector", 
            collection_id: Optional[str] = None) -> List[RetrievalResult]:
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
```

#### Internal Method Updates Required:
All internal retrieval methods need collection_id parameter:

```python
def _vector_retrieve(self, query: str, k: int, collection_id: Optional[str] = None) -> List[RetrievalResult]:
    """Vector-based retrieval with optional collection filtering."""
    return self.vector_db.search_similar(query, k=k, collection_id=collection_id)

def _keyword_retrieve(self, query: str, k: int, collection_id: Optional[str] = None) -> List[RetrievalResult]:
    """Keyword-based retrieval with optional collection filtering."""
    return self.vector_db.search_keyword(query, k=k, collection_id=collection_id)

def _hybrid_retrieve(self, query: str, k: int, collection_id: Optional[str] = None) -> List[RetrievalResult]:
    """Hybrid retrieval with optional collection filtering."""
    return self.vector_db.search_hybrid(query, k=k, collection_id=collection_id)
```

#### Import Addition:
```python
from typing import Optional  # if not already present
```

## Fix 4: Enhanced Experiment Result Export

### File: `main.py`
### Issue: _save_experiment_results function lacks config provenance
### Impact: MEDIUM - Results not traceable to parameters

#### Find and Replace the _save_experiment_results function:

#### New Implementation:
```python
def _save_experiment_results(results: List[ExperimentResult], output_path: str, 
                           experiment_metadata: Optional[Dict[str, Any]] = None):
    """Save experiment results with full configuration provenance."""
    
    # Prepare metadata
    metadata = experiment_metadata or {}
    metadata.update({
        'export_timestamp': datetime.now().isoformat(),
        'total_results': len(results),
        'claude_version': 'Experiment_v2_fixes'
    })
    
    # Format results with full config
    export_data = {
        'metadata': metadata,
        'results': []
    }
    
    for result in results:
        # Extract all configuration parameters
        config_dict = {}
        if hasattr(result, 'config') and result.config:
            config_dict = {
                'chunk_size': getattr(result.config, 'chunk_size', None),
                'chunk_overlap': getattr(result.config, 'chunk_overlap', None),
                'retrieval_k': getattr(result.config, 'retrieval_k', None),
                'temperature': getattr(result.config, 'temperature', None),
                'max_tokens': getattr(result.config, 'max_tokens', None),
                'profile': getattr(result.config, 'profile', None),
                'collection_id': getattr(result.config, 'collection_id', None),
                'retrieval_method': getattr(result.config, 'retrieval_method', 'vector')
            }
        
        # Create enhanced result entry
        result_dict = {
            'config': config_dict,
            'query': result.query,
            'response': result.response,
            'response_length_words': len(result.response.split()) if result.response else 0,
            'response_length_chars': len(result.response) if result.response else 0,
            'num_sources': result.num_sources,
            'created_at': result.created_at.isoformat() if result.created_at else None,
            'completed_at': result.completed_at.isoformat() if result.completed_at else None,
            'total_runtime': result.total_runtime,
            
            # Enhanced metrics from metadata
            'retrieval_time': result.metadata.get('retrieval_time') if result.metadata else None,
            'generation_time': result.metadata.get('generation_time') if result.metadata else None,
            'prompt_tokens': result.metadata.get('prompt_tokens') if result.metadata else None,
            'output_tokens': result.metadata.get('output_tokens') if result.metadata else None,
            'total_tokens': result.metadata.get('total_tokens') if result.metadata else None,
            'contexts_count': result.metadata.get('contexts_count') if result.metadata else None,
            'tokens_per_second': result.metadata.get('tokens_per_second') if result.metadata else None,
            'context_remaining': result.metadata.get('context_remaining') if result.metadata else None,
            
            # Full metadata for completeness
            'metadata': result.metadata if result.metadata else {}
        }
        
        export_data['results'].append(result_dict)
    
    # Save with proper formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(results)} experiment results to {output_path}")
    print(f"   Enhanced format includes full config provenance")
```

#### Required Import Addition:
```python
from datetime import datetime  # if not already present
```

## Fix 5: Experiment Configuration Class Extensions

### File: `src/experiment_templates.py` or relevant config file
### Issue: ExperimentConfig may need chunk parameter support
### Impact: LOW - Depends on current implementation

#### Verify ExperimentConfig includes:
```python
@dataclass
class ExperimentConfig:
    # Existing fields...
    
    # Chunking parameters (add if missing)
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    collection_id: Optional[str] = None
    
    # Other experimental parameters...
```

## Fix 6: VectorDatabase Collection Filtering Validation

### File: `src/vector_database.py`
### Issue: Ensure search methods properly handle collection_id parameter
### Impact: MEDIUM - Backend filtering needs verification

#### Verify Methods Support collection_id:
```python
def search_similar(self, query: str, k: int = 5, collection_id: Optional[str] = None) -> List[RetrievalResult]:
    """Vector similarity search with optional collection filtering."""
    # Implementation should filter by collection_id if provided
    
def search_keyword(self, query: str, k: int = 5, collection_id: Optional[str] = None) -> List[RetrievalResult]:
    """Keyword search with optional collection filtering."""
    # Implementation should filter by collection_id if provided
    
def search_hybrid(self, query: str, k: int = 5, collection_id: Optional[str] = None) -> List[RetrievalResult]:
    """Hybrid search with optional collection filtering."""
    # Implementation should filter by collection_id if provided
```

If methods don't support collection_id, add SQL WHERE clauses:
```sql
-- Example for vector search
WHERE collection_id = ? AND other_conditions...
```

## Fix 7: ReindexTool Collection Copy Functionality

### File: `src/reindex.py`
### Issue: May need source collection copying for experiments
### Impact: MEDIUM - Depends on rechunk_documents implementation

#### Potential Addition Needed:
```python
def copy_collection_documents(self, source_collection: str, target_collection: str) -> bool:
    """Copy documents from source to target collection before rechunking."""
    try:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Copy documents
            cursor.execute("""
                INSERT INTO documents (doc_id, filename, content, metadata_json, created_at, collection_id)
                SELECT doc_id, filename, content, metadata_json, created_at, ?
                FROM documents WHERE collection_id = ?
            """, (target_collection, source_collection))
            
            return True
    except Exception as e:
        self.logger.error(f"Failed to copy collection documents: {e}")
        return False
```

## Implementation Checklist

### Phase 1: Core Pipeline Fixes
- [ ] Update ExperimentRunner._create_rag_pipeline() with chunking support
- [ ] Add ExperimentRunner._ensure_chunked_collection() method
- [ ] Fix RAGPipeline.query() collection_id threading
- [ ] Update Retriever.retrieve() method signature and implementation
- [ ] Update all Retriever internal methods (_vector_retrieve, _keyword_retrieve, _hybrid_retrieve)

### Phase 2: Data Export and Configuration  
- [ ] Replace main.py _save_experiment_results() with enhanced version
- [ ] Verify/update ExperimentConfig dataclass with chunking parameters
- [ ] Add required imports to all modified files

### Phase 3: Backend Validation
- [ ] Verify VectorDatabase methods support collection_id filtering
- [ ] Test ReindexTool.rechunk_documents() with new collections
- [ ] Add collection copying functionality if needed

### Phase 4: Testing and Validation
- [ ] Create unit tests for collection ID threading
- [ ] Test end-to-end experiment run with chunking parameters
- [ ] Validate result exports include full configuration
- [ ] Test collection isolation (configs don't interfere)

## Risk Assessment

### High Risk Changes
1. **ExperimentRunner._create_rag_pipeline()**: Core experiment functionality
2. **Retriever method signatures**: Breaking changes to API
3. **Collection creation logic**: Database operations with failure potential

### Medium Risk Changes  
1. **Result export format**: May break existing analysis scripts
2. **VectorDatabase filtering**: Backend query changes

### Low Risk Changes
1. **Configuration class updates**: Additive changes
2. **Import additions**: Non-breaking additions

## Rollback Plan
1. **Git branch**: Create feature branch before changes
2. **Database backup**: Backup experimental databases before testing
3. **Configuration backup**: Save current system configuration
4. **Incremental testing**: Test each fix individually before combining

This comprehensive fix plan addresses all architectural issues that invalidated Experiment 1 and enables scientifically valid chunking experiments with proper parameter materialization and isolation.