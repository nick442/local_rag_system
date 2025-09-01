# Local RAG System - Complete Execution Plan for Coding Agent

## Phase 0: Repository Analysis Report (COMPLETED)

### Codebase Structure Overview

**Repository Path**: `/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/local_rag_system`

#### Core Source Modules (`src/`)
1. **Document Processing**
   - `document_ingestion.py` - Loaders for text/PDF/HTML/Markdown, chunking with tiktoken
   - `deduplication.py` - Content hash & LSH-based duplicate detection
   - `corpus_manager.py` - Parallel bulk ingestion with checkpointing

2. **Vector & Retrieval**
   - `vector_database.py` - SQLite + sqlite-vec extension, FTS5 integration
   - `embedding_service.py` - SentenceTransformers with GPU/MPS acceleration
   - `retriever.py` - Vector/keyword/hybrid retrieval methods

3. **RAG Pipeline**
   - `rag_pipeline.py` - Main orchestration, query/chat modes
   - `prompt_builder.py` - Template-based prompt construction
   - `llm_wrapper.py` - llama-cpp-python with Metal acceleration

4. **Configuration & Management**
   - `config_manager.py` - ProfileConfig (6 params) + ExperimentConfig (60+ params)
   - `system_manager.py` - Component initialization and health checks
   - `cli_chat.py` - Interactive chat interface

5. **Analytics & Optimization**
   - `corpus_analytics.py` - Quality metrics, clustering, statistics
   - `corpus_organizer.py` - Collection management
   - `reindex.py` - Database maintenance tools
   - `experiment_runner.py` - Parameter sweep orchestration
   - `experiment_templates.py` - Pre-defined experiment configurations

#### Configuration State
- **Current**: Unified `config/rag_config.yaml` with 3 profiles (fast/balanced/quality)
- **PDF Spec Alignment**: Partial - profiles exist but don't propagate to components

#### Identified Gaps vs PDF Spec

1. **Configuration Issues**
   - ✅ Config consolidation done (single YAML)
   - ❌ Profile parameters not propagating to pipeline components
   - ❌ SystemManager still exists alongside simpler factories
   - ❌ Multiple initialization paths remain

2. **Model & Resource Management**
   - ❌ Models reloaded on each command (memory waste)
   - ❌ No model caching/reuse across operations
   - ❌ Missing embedding dimension handshake
   - ❌ No PRAGMA tuning for SQLite

3. **Interfaces & Abstractions**
   - ❌ No pluggable retrieval interface
   - ❌ No chunker interface abstraction
   - ❌ Vector index tightly coupled to SQLite
   - ❌ Missing query reformulation integration

4. **Metrics & Logging**
   - ✅ Basic metrics captured in pipeline
   - ❌ No JSONL structured logging
   - ❌ Metrics not exposed in standardized format
   - ❌ Missing detailed retrieval vs generation timing

5. **CLI & Batch Processing**
   - ✅ Basic CLI exists with commands
   - ❌ No batch experiment CLI as specified
   - ❌ Profile switching doesn't reload components
   - ❌ Collection filtering not threaded through

6. **Documentation & Testing**
   - ✅ Component docs exist in `docs/`
   - ❌ Not organized as Core Dev Kit / Experiment Kit
   - ❌ Tests don't cover new abstractions
   - ❌ Missing migration guide for legacy docs

---

## Phase 1: Configuration & Initialization Consolidation

**Why**: PDF spec mandates single config source. Currently have overlapping SystemManager/ConfigManager/direct params.

### Actions
1. **Remove SystemManager redundancy**
   ```python
   # File: src/system_manager.py
   # Action: DELETE entire file
   # Reason: Duplicates ConfigManager, adds unnecessary indirection
   ```

2. **Update main.py to use ConfigManager directly**
   ```python
   # File: main.py
   # Modify CLI initialization to load ConfigManager once
   # Pass config dict to all components
   # Remove SystemManager imports and usage
   ```

3. **Extend RAGPipeline to accept profile config**
   ```python
   # File: src/rag_pipeline.py
   # Add profile_config parameter to __init__
   # Apply retrieval_k, chunk_size, etc. from config
   ```

### Files Touched
- `main.py` - Remove SystemManager, use ConfigManager
- `src/system_manager.py` - DELETE
- `src/rag_pipeline.py` - Accept ProfileConfig
- `src/cli_chat.py` - Use ConfigManager for profile switching

### Exit Criteria
- Single YAML config loads once per session
- Profile parameters propagate to all components
- No SystemManager references remain

### Tests
```bash
python main.py query "test" --profile fast
# Verify retrieval_k=3 is used
python main.py chat --profile quality  
# Verify retrieval_k=10 is used
```

---

## Phase 2: Model Resource Management

**Why**: PDF spec requires model reuse. Currently reloading models wastes 4GB+ RAM per command.

### Actions
1. **Create ModelCache singleton**
   ```python
   # File: src/model_cache.py (NEW)
   # Implement lazy-loaded singleton for embedding & LLM models
   # Thread-safe access with locks
   ```

2. **Update components to use cache**
   ```python
   # File: src/embedding_service.py
   # Check ModelCache before loading
   # File: src/llm_wrapper.py  
   # Check ModelCache before loading
   ```

3. **Add embedding dimension handshake**
   ```python
   # File: src/vector_database.py
   # Validate embedding_dimension matches model
   # Raise error if mismatch
   ```

### Files Touched
- `src/model_cache.py` - NEW singleton cache
- `src/embedding_service.py` - Use cache
- `src/llm_wrapper.py` - Use cache
- `src/vector_database.py` - Dimension validation
- `src/rag_pipeline.py` - Use cached models

### Exit Criteria
- Models load once per session
- Memory usage stable across commands
- Dimension mismatches caught early

### Tests
```python
# File: tests/test_model_cache.py (NEW)
# Test singleton behavior
# Test thread safety
# Test dimension validation
```

---

## Phase 3: Pluggable Interfaces

**Why**: PDF spec requires abstraction for retrieval, chunking, vector indices. Currently tightly coupled.

### Actions
1. **Create retrieval interface**
   ```python
   # File: src/interfaces/retrieval_interface.py (NEW)
   # Abstract base class for retrievers
   # Implementations: DefaultRetriever, FaissRetriever, etc.
   ```

2. **Create chunker interface**
   ```python
   # File: src/interfaces/chunker_interface.py (NEW)
   # Abstract base class for chunkers
   # Implementations: TokenChunker, SemanticChunker, etc.
   ```

3. **Create vector index interface**
   ```python
   # File: src/interfaces/vector_index_interface.py (NEW)
   # Abstract base class for vector stores
   # Implementations: SqliteVecIndex, FaissIndex, etc.
   ```

4. **Refactor components to use interfaces**
   ```python
   # Update retriever.py, document_ingestion.py, vector_database.py
   # to work with interfaces
   ```

### Files Touched
- `src/interfaces/` - NEW directory
- `src/interfaces/retrieval_interface.py` - NEW
- `src/interfaces/chunker_interface.py` - NEW
- `src/interfaces/vector_index_interface.py` - NEW
- `src/retriever.py` - Use interface
- `src/document_ingestion.py` - Use interface
- `src/vector_database.py` - Use interface

### Exit Criteria
- Can swap retrieval methods via config
- Can swap chunking strategies via config
- Can swap vector indices via config

### Tests
```python
# File: tests/test_interfaces.py (NEW)
# Test interface implementations
# Test swappability
```

---

## Phase 4: Metrics & Logging Enhancement

**Why**: PDF spec requires JSONL structured logging with detailed metrics.

### Actions
1. **Create metrics collector**
   ```python
   # File: src/metrics.py (NEW)
   # Collect timing, token counts, scores
   # Export as JSONL
   ```

2. **Instrument pipeline components**
   ```python
   # Add metrics.track() calls throughout
   # Capture retrieval_time, generation_time separately
   ```

3. **Add PRAGMA optimizations**
   ```python
   # File: src/vector_database.py
   # Add PRAGMA journal_mode=WAL
   # Add PRAGMA synchronous=NORMAL
   # Add PRAGMA cache_size=-64000
   ```

### Files Touched
- `src/metrics.py` - NEW collector
- `src/rag_pipeline.py` - Add metrics tracking
- `src/retriever.py` - Track retrieval timing
- `src/llm_wrapper.py` - Track generation timing
- `src/vector_database.py` - Add PRAGMAs
- `logs/metrics.jsonl` - NEW output file

### Exit Criteria
- All operations emit JSONL metrics
- Retrieval vs generation time separated
- SQLite optimized with PRAGMAs

### Tests
```bash
python main.py query "test" --metrics
# Check logs/metrics.jsonl for structured output
```

---

## Phase 5: CLI Batch & Experiment Runner

**Why**: PDF spec requires batch CLI for experiments. Current runner doesn't apply chunk params.

### Actions
1. **Fix experiment runner parameter application**
   ```python
   # File: src/experiment_runner.py
   # Apply chunk_size, chunk_overlap from config
   # Thread collection_id through stack
   ```

2. **Add batch CLI command**
   ```python
   # File: main.py
   # Add 'experiment batch' command
   # Accept query file, config overrides
   # Output JSONL results
   ```

3. **Fix collection filtering**
   ```python
   # File: src/rag_pipeline.py
   # Pass collection_id to retriever
   # File: src/retriever.py
   # Forward collection_id to vector_database
   ```

### Files Touched
- `src/experiment_runner.py` - Fix parameter application
- `main.py` - Add batch command
- `src/rag_pipeline.py` - Thread collection_id
- `src/retriever.py` - Forward collection_id
- `src/vector_database.py` - Already supports collection_id

### Exit Criteria
- Chunk parameters actually change behavior
- Collection filtering works end-to-end
- Batch CLI produces JSONL output

### Tests
```bash
python main.py experiment batch --queries test_data/queries.json --profile fast
# Verify chunk_size=256 is applied
# Verify results in JSONL format
```

---

## Phase 6: Documentation Reorganization

**Why**: PDF spec requires Core Dev Kit and Experiment Runner Kit documentation.

### Actions
1. **Create Core Dev Kit**
   ```bash
   mkdir docs/core
   # Create: ARCH.md, CONFIG.md, RETRIEVAL.md, METRICS.md, DEV_GUIDE.md
   # Each ≤400 words
   ```

2. **Create Experiment Runner Kit**
   ```bash
   mkdir docs/experiments  
   # Create: EXPERIMENTS.md, EVAL.md, RESULTS.md, TEMPLATES/
   # Each ≤400 words
   ```

3. **Archive legacy docs**
   ```bash
   mkdir docs/archive
   git mv docs/*.md docs/archive/
   # Except new kit docs
   ```

### Files Touched
- `docs/core/` - NEW directory with 5 docs
- `docs/experiments/` - NEW directory with 4 docs
- `docs/archive/` - Moved legacy docs

### Exit Criteria
- Core Dev Kit complete
- Experiment Kit complete
- Legacy docs archived

---

## Phase 7: Testing & Validation

**Why**: Ensure all changes work correctly on Mac mini M4 16GB.

### Actions
1. **Create integration tests**
   ```python
   # File: tests/test_integration.py (NEW)
   # Test config propagation
   # Test model caching
   # Test interface swapping
   ```

2. **Create performance benchmarks**
   ```python
   # File: tests/test_performance.py (NEW)
   # Measure memory usage
   # Measure query latency
   # Verify 16GB constraint
   ```

3. **Update existing tests**
   ```python
   # Fix broken imports
   # Update for new interfaces
   ```

### Files Touched
- `tests/test_integration.py` - NEW
- `tests/test_performance.py` - NEW
- `tests/test_*.py` - Update existing

### Exit Criteria
- All tests pass
- Memory usage <16GB
- Query latency acceptable

---

## PR Breakdown

### PR 1: Configuration Consolidation
- Remove SystemManager
- Update main.py
- Fix config propagation
- **Files**: 4 changes

### PR 2: Model Resource Management
- Add ModelCache
- Update services to use cache
- Add dimension validation
- **Files**: 5 changes

### PR 3: Pluggable Interfaces
- Create interface abstractions
- Refactor to use interfaces
- **Files**: 7 changes

### PR 4: Metrics & Logging
- Add metrics collector
- Instrument components
- Add SQLite PRAGMAs
- **Files**: 6 changes

### PR 5: CLI & Experiment Fixes
- Fix experiment runner
- Add batch CLI
- Fix collection filtering
- **Files**: 5 changes

### PR 6: Documentation
- Create Core Dev Kit
- Create Experiment Kit
- Archive legacy docs
- **Files**: 15+ new docs

### PR 7: Testing
- Add integration tests
- Add performance tests
- Update existing tests
- **Files**: 10+ test files

---

## Definition of Done

✅ **Repository Scanned**: All Python modules analyzed, gaps identified
✅ **PDF Specs Implemented**: All 10 recommendations from PDF
✅ **Documentation Complete**: Core Dev Kit + Experiment Kit ready
✅ **Tests Passing**: Integration, performance, unit tests green
✅ **Memory Stable**: <16GB usage confirmed
✅ **CLI Enhanced**: Batch experiments working
✅ **Legacy Archived**: Old docs moved to archive/

---

## Immediate Next Steps for Agent

1. **Start with PR 1**: Configuration consolidation is foundation
2. **Test after each PR**: Ensure nothing breaks
3. **Document changes**: Update relevant docs with each PR
4. **Benchmark regularly**: Track memory and performance

---

## Notes on Current State

- **Working**: Basic RAG pipeline, ingestion, retrieval, chat
- **Partially Working**: Experiments (params don't apply), profiles (don't propagate)
- **Missing**: Model caching, interfaces, JSONL metrics, batch CLI
- **Quality**: Code is well-structured but has accidental complexity

This plan provides atomic, testable changes that gradually transform the codebase to match the PDF specification while maintaining functionality throughout.