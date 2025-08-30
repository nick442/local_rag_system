# Vector Database

## Overview
The Vector Database (`src/vector_database.py`) provides high-performance vector similarity search using SQLite with the sqlite-vec extension, enabling sub-millisecond similarity queries on large document collections.

## Core Classes

### VectorDatabase
**Purpose**: Main database interface for document and embedding storage with vector search capabilities
**Key Features**:
- **Vector Search**: sqlite-vec extension for efficient similarity search
- **Document Storage**: Structured storage for documents, chunks, and metadata
- **Hybrid Search**: Combines vector similarity with keyword and metadata filtering
- **Performance Optimization**: Proper indexing and query optimization
- **Data Integrity**: ACID transactions and foreign key constraints

**Database Schema**:
```sql
-- Documents table
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  source_path TEXT NOT NULL,
  ingested_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  content_hash TEXT,
  file_size INTEGER,
  total_chunks INTEGER DEFAULT 0,
  collection_id TEXT NOT NULL DEFAULT 'default'
);

-- Document chunks table  
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  token_count INTEGER NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  collection_id TEXT NOT NULL DEFAULT 'default',
  FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
);

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id TEXT PRIMARY KEY,
  embedding_vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE
);

-- Full‑text search (FTS5) for keyword queries
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED,
  content,
  content='chunks',
  content_rowid='rowid'
);

-- Triggers to sync FTS index with chunks
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(chunk_id, content) VALUES (new.chunk_id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, chunk_id, content) VALUES('delete', old.chunk_id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, chunk_id, content) VALUES('delete', old.chunk_id, old.content);
  INSERT INTO chunks_fts(chunk_id, content) VALUES (new.chunk_id, new.content);
END;

-- Optional: sqlite‑vec vector table (when extension is available)
-- CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_vec USING vec0(
--   chunk_id TEXT PRIMARY KEY,
--   embedding float[384]
-- );
```

## Key Methods

### insert_document(doc_id: str, source_path: str, metadata: Dict) -> bool
**Purpose**: Insert a document record into the database
**Features**:
- Duplicate detection (returns False if doc_id exists)
- Metadata JSON serialization
- Automatic timestamp generation
- Content hash storage for deduplication

**Inputs**: 
- `doc_id`: Unique document identifier
- `source_path`: Original file path
- `metadata`: Document metadata dictionary

**Outputs**: Boolean indicating successful insertion

### insert_chunk(chunk: DocumentChunk, embedding: np.ndarray) -> bool
**Purpose**: Store a document chunk with its vector embedding
**Features**:
- Dual storage: relational (chunks) + vector (embeddings_vec)  
- Embedding validation (dimension checking)
- Automatic ID generation if not provided
- Transaction safety

**Inputs**:
- `chunk`: DocumentChunk object with content and metadata
- `embedding`: NumPy array with vector embedding (384-dim)

**Outputs**: Boolean indicating successful storage

### search_similar(query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> List[Dict]
**Purpose**: Find most similar chunks using vector similarity search
**Features**:
- **sqlite-vec Integration**: Sub-millisecond search performance
- **Similarity Thresholding**: Filter results by minimum similarity
- **Metadata Enrichment**: Returns chunks with document context
- **Fallback Search**: Manual search if sqlite-vec unavailable

**Algorithm**:
1. Query sqlite-vec virtual table for nearest neighbors
2. Join with chunks and documents tables for full context
3. Apply similarity threshold filtering
4. Return ranked results with similarity scores

**Performance**: 
- **With sqlite-vec**: <1ms for 10k documents, <10ms for 100k documents
- **Fallback mode**: 100ms-1s depending on corpus size

### keyword_search(query: str, k: int = 5) -> List[Dict]
**Purpose**: Full-text search using SQLite FTS (Full-Text Search)
**Features**:
- **FTS5 Integration**: Fast full-text search with ranking
- **Boolean Operators**: AND, OR, NOT support
- **Phrase Matching**: Exact phrase search with quotes
- **Stemming**: Automatic word stem matching

**Query Examples**:
- Simple: `"machine learning"`
- Boolean: `"machine AND learning"`  
- Phrase: `"\"artificial intelligence\""`
- Exclusion: `"AI NOT robot"`

### hybrid_search(query_embedding: np.ndarray, query_text: str, k: int = 5, vector_weight: float = 0.7) -> List[Dict]
**Purpose**: Combine vector similarity and keyword search for best results
**Features**:
- **Score Fusion**: Weighted combination of similarity and text relevance
- **Normalization**: Scores normalized to [0,1] range before combination
- **Reranking**: Final ranking based on combined scores
- **Fallback**: Graceful degradation if one search method fails

**Algorithm**:
1. Perform vector similarity search
2. Perform keyword search in parallel
3. Normalize scores from both methods
4. Combine using weighted formula: `vector_weight * sim_score + (1-vector_weight) * text_score`
5. Rerank and return top-k results

## Performance Optimization

### sqlite-vec Extension Loading
**Role**: Provides native vector operations in SQLite for fast ANN search

**Loading Order**:
- Try Python package: `import sqlite_vec; sqlite_vec.load(conn)`
- Fallback to vendor dylib: `vendor/sqlite-vec/vec0.dylib` (if present)
- Otherwise use manual cosine similarity (O(n)) as a safe fallback

**Environment Flags**:
- `RAG_DISABLE_SQLITE_VEC_VENDOR=1` — disable vendor dylib loading
- `RAG_SQLITE_VEC_TRY_VENDOR=0` — also disables vendor attempt

**Verification**:
```python
db = VectorDatabase("test.db")
# Logs indicate: loaded via Python package, vendor dylib, or fallback
```

### Index Optimization
- **Vector Index**: sqlite-vec automatically creates vector indices
- **Text Index**: FTS5 indices for keyword search
- **Foreign Keys**: Proper relationships between tables
- **Query Optimization**: Efficient JOIN operations

### Memory Management
- **Connection Pooling**: Reuses database connections
- **Embedding Storage**: Efficient BLOB storage for vectors
- **Batch Operations**: Transaction batching for bulk inserts
- **Cache Management**: Query result caching where appropriate

## Search Methods Comparison

| Method | Use Case | Performance | Accuracy | Best For |
|--------|----------|-------------|-----------|-----------|
| Vector Search | Semantic similarity | <1ms | High for concepts | "What is AI?" |
| Keyword Search | Exact term matching | <5ms | High for terms | "neural network architecture" |  
| Hybrid Search | Best of both | <10ms | Highest overall | Most real-world queries |

### When to Use Each Method

**Vector Search**:
- Semantic queries ("concepts similar to...")
- Multilingual search
- Concept discovery
- When exact terms may not appear in documents

**Keyword Search**: 
- Specific term lookup
- Technical documentation search
- When you know exact terminology
- Boolean query requirements

**Hybrid Search**:
- General question answering
- Unknown query types
- Maximum recall requirements
- Production RAG systems

## Integration Points

### Input Sources
- **Corpus Manager**: Bulk embedding generation during ingestion
- **RAG Pipeline**: Query embedding for real-time search
- **Analytics**: Similarity analysis and clustering
- **Deduplication**: Semantic duplicate detection

### Output Destinations
- **Search Results**: Ranked document chunks with similarity scores
- **Analytics Reports**: Embedding-based corpus insights
- **Recommendation Systems**: Similar document suggestions
- **Quality Assessment**: Embedding coverage and distribution analysis

## Configuration Examples

### High-Performance Setup
```python
# Optimized for speed
db = VectorDatabase(
    db_path="vectors_optimized.db",
    embedding_dimension=384  # Smaller, faster embeddings
)
```

### High-Quality Setup  
```python
# Optimized for accuracy
db = VectorDatabase(
    db_path="vectors_quality.db", 
    embedding_dimension=768  # Larger, more accurate embeddings
)
```

### Development Setup
```python
# Memory-efficient for testing
db = VectorDatabase(
    db_path=":memory:",  # In-memory for tests
    embedding_dimension=384
)
```

## Error Handling and Recovery

### Common Issues
- **sqlite-vec Not Found**: Falls back to manual similarity search
- **Dimension Mismatch**: Validates embedding dimensions before insert
- **Corrupted Database**: Provides repair and validation tools
- **Memory Overflow**: Automatic batch size adjustment

### Diagnostic Methods
- **get_database_stats()**: Comprehensive database statistics
- **validate_integrity()**: Check for missing or corrupted data
- **performance_test()**: Benchmark search operations

## Database Statistics

`get_database_stats()` returns basic metrics:
```python
{
  'documents': 1500,
  'chunks': 12500,
  'embeddings': 12500,
  'database_size_bytes': 275251200,
  'database_size_mb': 262.56,
  'embedding_dimension': 384
}
```

## Notes and Limitations
- Collection scoping is available for vector search via a `collection_id` filter. Keyword and hybrid paths may not apply the collection filter in the current implementation.
- The `embeddings` table does not store `collection_id`; scoping is enforced using `chunks.collection_id` via joins.

## Advanced Features

### Custom Similarity Functions
Support for different similarity metrics beyond cosine similarity:
- Cosine similarity (default)
- Euclidean distance  
- Dot product similarity
- Angular distance

### Metadata Filtering
Combine vector search with metadata constraints:
```python
# Search within specific document types
results = db.search_similar(
    query_embedding,
    k=10,
    metadata_filter={'file_type': '.pdf'}
)
```

### Bulk Operations
Optimized methods for large-scale operations:
- `bulk_insert_embeddings()`: Efficient batch insertion
- `rebuild_indices()`: Index reconstruction and optimization
- `vacuum_database()`: Space reclamation and defragmentation
