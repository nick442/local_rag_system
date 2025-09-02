# JSON Pipeline Process Documentation

## Overview
This document describes how the JSON document pipeline is **supposed to work** for converting JSON documents to vector database entries. The system repeatedly gets stuck at 7 documents during ingestion, requiring investigation of the complete pipeline.

## Pipeline Architecture

### Expected Flow
```
JSON Documents → Document Ingestion → Chunking → Embedding Generation → Vector Database Storage
```

## Core Components Used

### 1. CorpusManager (`src/corpus_manager.py`)
**Purpose**: Orchestrates the entire ingestion pipeline with parallel processing and checkpointing.

**Key Method**: `ingest_directory()` (async)
- Scans directory for files using glob patterns
- Handles deduplication based on content hashes
- Manages parallel processing with ThreadPoolExecutor
- Implements checkpointing every N files for resume capability
- Calls `process_single_document()` for each file

**Key Method**: `process_single_document(file_path, collection_id)`
- Generates unique document ID (UUID)
- Calls DocumentIngestionService to load and chunk document
- Generates embeddings via EmbeddingService  
- Stores document and chunks in VectorDatabase
- Returns success/failure status with statistics

### 2. DocumentIngestionService (`src/document_ingestion.py`)
**Purpose**: Loads and parses documents from multiple formats.

**Key Method**: `ingest_document(file_path)`
- Detects file type based on extension
- Uses appropriate loader (TextLoader, PDFLoader, HTMLLoader, MarkdownLoader)
- Chunks document using DocumentChunker
- Returns List[DocumentChunk] objects

**Expected Formats**:
- `.txt`: Plain text files
- `.pdf`: PDF documents  
- `.html`: HTML files
- `.md`: Markdown files

### 3. DocumentChunker (`src/document_ingestion.py`)
**Purpose**: Splits documents into manageable chunks for embedding.

**Configuration**:
- Default chunk size: 1000 characters
- Default overlap: 200 characters
- Preserves sentence boundaries when possible

**Output**: DocumentChunk objects with:
- `content`: Text content
- `metadata`: Source information
- `token_count`: Estimated tokens
- `doc_id`: Parent document ID (set by CorpusManager)
- `chunk_id`: Unique chunk identifier
- `chunk_index`: Position within document

### 4. EmbeddingService (`src/embedding_service.py`)
**Purpose**: Generates vector embeddings from text using sentence-transformers.

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Device: MPS (Apple Silicon GPU acceleration)
- Batch processing for efficiency

**Key Method**: `embed_texts(texts: List[str])`
- Processes texts in batches
- Returns numpy arrays of embeddings
- Handles memory management and device optimization

### 5. VectorDatabase (`src/vector_database.py`)
**Purpose**: Stores documents, chunks, and embeddings with sqlite-vec for fast similarity search.

**Schema**:
```sql
documents (doc_id, source_path, ingested_at, metadata_json, content_hash, file_size, collection_id)
chunks (chunk_id, doc_id, chunk_index, content, token_count, metadata_json, created_at, collection_id)
embeddings (chunk_id, embedding_vector, created_at)
embeddings_vec (chunk_id, embedding) -- sqlite-vec virtual table
```

**Key Methods**:
- `insert_document(doc_id, source_path, metadata, collection_id)`
- `insert_chunk(chunk, embedding, collection_id)`

## Expected Processing Steps

### Step 1: Directory Scan
```python
files = manager.scan_directory(path, "*.txt")
# Should find all .txt files in directory
```

### Step 2: Deduplication
```python
unique_files, duplicates = manager.check_duplicates(files)
# Creates SHA-256 hashes to identify duplicates
```

### Step 3: Parallel Processing Setup
```python
with ThreadPoolExecutor(max_workers=1) as executor:
    # Submit process_single_document tasks
    future_to_file = {
        executor.submit(manager.process_single_document, file_path, collection_id): file_path
        for file_path in files
    }
```

### Step 4: Per-Document Processing
For each document:
```python
# 1. Load document content
chunks = ingestion_service.ingest_document(file_path)

# 2. Set chunk metadata
for i, chunk in enumerate(chunks):
    chunk.doc_id = document_id
    chunk.chunk_index = i
    chunk.chunk_id = f"{document_id}_chunk_{i}"

# 3. Generate embeddings
chunk_texts = [chunk.content for chunk in chunks]
embeddings = embedding_service.embed_texts(chunk_texts)

# 4. Store in database
db.insert_document(document_id, str(file_path), metadata, collection_id)
for chunk, embedding in zip(chunks, embeddings):
    db.insert_chunk(chunk, embedding, collection_id)
```

### Step 5: Progress Tracking
- Progress bar updates after each document
- Checkpoints saved every 5 documents
- Statistics accumulated (files_processed, chunks_created, etc.)

## Known Issues Requiring Investigation

### 1. Consistent Hang at 7 Documents
**Symptom**: Processing stops after exactly 7 documents across multiple test runs
**Potential Causes**:
- Database connection/transaction issues
- Memory leak or resource exhaustion
- Threading/async coordination problems
- Specific document content causing parser to hang
- Database locking or sqlite-vec extension issues

### 2. Performance Discrepancy
**Symptom**: Initial progress shows 9-10 docs/sec, actual measured rate is 0.35 docs/sec
**Likely Cause**: Initial progress bars measure different operations (file scan, dedup) not actual processing

### 3. Silent Failures
**Symptom**: Process appears to run (consumes CPU) but makes no database progress
**Investigation Needed**: Check if process hangs in specific component

## Investigation Strategy

### 1. Component Isolation
Test each component independently:
- DocumentIngestionService with single file
- EmbeddingService with single text  
- VectorDatabase insert operations
- CorpusManager without parallel processing

### 2. Document-Level Analysis
- Check which specific document (#8) causes the hang
- Analyze document content for parsing issues
- Test with documents of different sizes/formats

### 3. Database Investigation
- Check for transaction locks
- Verify sqlite-vec extension stability
- Monitor database file size growth
- Check for connection pool exhaustion

### 4. Threading Analysis
- Test with max_workers=1 (already done - still hangs)
- Check ThreadPoolExecutor resource cleanup
- Monitor thread creation/destruction

### 5. Memory Profiling
- Monitor memory usage during processing
- Check for memory leaks in embedding service
- Verify garbage collection of large objects

## Expected Performance Targets

For production readiness:
- **Target Rate**: 5-10 documents per second
- **200 documents**: Should complete in 20-40 seconds  
- **11k documents**: Should complete in 18-37 minutes
- **Memory Usage**: Should remain stable throughout processing
- **Error Rate**: <1% failed documents

## Current Status

**Working**: Collection assignment fix completed - documents properly stored in specified collections
**Broken**: Processing pipeline hangs consistently after 7 documents
**Next Steps**: Fresh investigation needed to identify root cause of hang

## File Locations

- Test data: `data/test_200/` (200 .txt files)
- Performance test: `test_performance.py`
- Log file: `performance_test.log`
- Core components: `src/corpus_manager.py`, `src/document_ingestion.py`, `src/embedding_service.py`, `src/vector_database.py`

This pipeline should work reliably for production-scale document ingestion once the hang issue is resolved.