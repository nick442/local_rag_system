# Phase 4: RAG Pipeline Components Implementation

## Context Loading
**FIRST STEP - MANDATORY**: Read the handoff file from previous phases:
```bash
cat handoff/phases_1_3_complete.json
```
This contains all project paths, installed packages, and model locations you need.

## Your Mission
Implement the core RAG pipeline components: document ingestion, embedding generation, vector database schema, and retrieval module. You are building the data processing backbone of the RAG system.

## Prerequisites Check
1. Verify you're in the correct project directory (check the `project_structure.root` from handoff)
2. Activate conda environment: `conda activate rag_env`
3. Verify llama-cpp-python is installed: `python -c "import llama_cpp; print('OK')"`
4. Verify sentence-transformers: `python -c "from sentence_transformers import SentenceTransformer; print('OK')"`

## Implementation Tasks

### Task 4.1: Document Ingestion Module
Create `src/document_ingestion.py`:

```python
# Required functionality:
# 1. DocumentLoader base class with abstract load() method
# 2. PDFLoader class using PyPDF2
# 3. HTMLLoader class using beautifulsoup4 and html2text
# 4. MarkdownLoader class
# 5. TextLoader for plain text files
# 
# Each loader must:
# - Extract text content
# - Preserve metadata (filename, page numbers, headers)
# - Return standardized Document objects
# 
# Document class structure:
# - content: str (the text)
# - metadata: dict (source, page, type, timestamp)
# - doc_id: str (unique identifier)
```

Key requirements:
- Chunk documents into 512 tokens with 128 token overlap
- Use tiktoken for accurate token counting
- Generate deterministic document IDs using hash of content
- Store original document path in metadata
- Handle encoding errors gracefully

### Task 4.2: Embedding Pipeline
Create `src/embedding_service.py`:

```python
# Required functionality:
# 1. EmbeddingService class
# 2. Load model from models/embeddings/ directory (path in handoff)
# 3. Batch processing (32 documents at a time)
# 4. Progress bar using tqdm for large batches
# 5. Async support for concurrent processing
# 6. Memory-efficient processing (clear cache between batches)
```

Implementation notes:
- Model path is in handoff JSON under `models.embeddings.path`
- Use MPS device if available: `device = "mps" if torch.backends.mps.is_available() else "cpu"`
- Normalize embeddings to unit vectors
- Return embeddings as numpy arrays

### Task 4.3: Vector Database Schema
Create `src/vector_database.py`:

```python
# SQLite schema with sqlite-vec extension
# Tables needed:
# 1. documents (doc_id, source_path, ingested_at, metadata_json)
# 2. chunks (chunk_id, doc_id, chunk_index, content, token_count, metadata_json)
# 3. embeddings (chunk_id, embedding_vector)
# 
# Required methods:
# - init_database(): Create tables and load sqlite-vec
# - insert_document(): Add document record
# - insert_chunk(): Add chunk with embedding
# - search_similar(): Vector similarity search
# - hybrid_search(): Combine vector + keyword search
```

Critical implementation details:
- Load sqlite-vec extension: `conn.load_extension("sqlite-vec")`
- Use HNSW index for vector search
- Store embeddings as BLOB (numpy.tobytes())
- Implement cosine similarity search
- Add FTS5 for keyword search capability

### Task 4.4: Retrieval Module
Create `src/retriever.py`:

```python
# Required functionality:
# 1. Retriever class wrapping vector database
# 2. retrieve(query, k=5): Get top-k similar chunks
# 3. retrieve_with_context(): Include surrounding chunks
# 4. filter_by_metadata(): Source/date filtering
# 5. assemble_context(): Build prompt context from chunks
```

Context assembly requirements:
- Order chunks by relevance score
- Include chunk metadata in context
- Deduplicate overlapping content
- Format as: `[Source: {filename}]\n{content}\n\n`
- Maximum context length: 6000 tokens

## Testing Requirements
Create `test_phase_4.py`:
1. Test document loading for each file type
2. Test chunking with overlap
3. Test embedding generation
4. Test vector storage and retrieval
5. Test end-to-end: ingest → embed → store → retrieve

## Output Requirements
Create `handoff/phase_4_complete.json` with:
```json
{
  "timestamp": "ISO-8601 timestamp",
  "phase": 4,
  "created_files": [
    "src/document_ingestion.py",
    "src/embedding_service.py", 
    "src/vector_database.py",
    "src/retriever.py",
    "test_phase_4.py"
  ],
  "database": {
    "path": "data/rag_vectors.db",
    "schema_version": "1.0",
    "tables": ["documents", "chunks", "embeddings"]
  },
  "capabilities": {
    "supported_formats": ["pdf", "html", "markdown", "txt"],
    "chunk_size": 512,
    "chunk_overlap": 128,
    "embedding_dimensions": 384,
    "max_batch_size": 32
  },
  "test_results": {
    "all_tests_passed": true,
    "test_output": "output from pytest"
  },
  "usage_example": "python -c 'from src.retriever import Retriever; r = Retriever(); results = r.retrieve(\"test query\")'"
}
```

## Validation Checklist
Before marking complete:
- [ ] All four modules implemented and importable
- [ ] SQLite database created with schema
- [ ] Can ingest a sample PDF/HTML/Markdown file
- [ ] Can generate embeddings for text
- [ ] Can retrieve similar chunks given a query
- [ ] All tests pass
- [ ] Handoff file created with all required fields

## Common Issues to Avoid
1. Don't use network-dependent models - use local cached models only
2. Ensure all file paths are relative to project root
3. Handle missing sqlite-vec extension gracefully
4. Don't load entire embedding model repeatedly - cache it
5. Implement proper error handling for malformed documents

Remember: The next phase depends on your retrieval module working correctly. Test thoroughly!