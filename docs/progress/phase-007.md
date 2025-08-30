# Phase 7: Corpus Management Implementation

## Context Loading
**FIRST STEP - MANDATORY**: Read previous handoff files:
```bash
cat handoff/phase_4_complete.json  # Document ingestion capabilities
cat handoff/phase_5_complete.json  # Pipeline integration
cat handoff/phase_6_complete.json  # CLI structure
```

## Your Mission
Build comprehensive corpus management tools for ingesting, organizing, and maintaining document collections. This phase enables users to build and manage their knowledge bases.

## Prerequisites Check
1. Verify ingestion modules: `python -c "from src.document_ingestion import PDFLoader, HTMLLoader; print('OK')"`
2. Verify embedding service: `python -c "from src.embedding_service import EmbeddingService; print('OK')"`
3. Verify database: Check if `data/rag_vectors.db` exists

## Implementation Tasks

### Task 7.1: Bulk Ingestion Pipeline
Create `src/corpus_manager.py`:

```python
# Bulk document processing system:
# 1. Recursive directory scanning
# 2. Parallel document processing
# 3. Progress tracking with tqdm
# 4. Duplicate detection
# 5. Incremental updates
```

Core implementation:
```python
import asyncio
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.document_ingestion import DocumentLoader
from src.embedding_service import EmbeddingService
from src.vector_database import VectorDatabase

class CorpusManager:
    def __init__(self, db_path="data/rag_vectors.db"):
        self.db = VectorDatabase(db_path)
        self.embedding_service = EmbeddingService()
        
    async def ingest_directory(self, path, pattern="**/*"):
        # 1. Scan for documents
        # 2. Filter by supported types
        # 3. Check for duplicates
        # 4. Process in batches
        # 5. Show progress bar
        
    def process_document(self, file_path):
        # 1. Load document
        # 2. Chunk content
        # 3. Generate embeddings
        # 4. Store in database
        # 5. Return statistics
```

Required features:
- Support glob patterns: `*.pdf`, `**/*.md`
- Parallel processing (use multiprocessing)
- Checkpointing for resume on failure
- Dry-run mode to preview changes
- Statistics reporting (files processed, chunks created)

### Task 7.2: Corpus Organization
Create `src/corpus_organizer.py`:

```python
# Corpus organization by collections/namespaces:
# 1. Create named collections
# 2. Tag documents with metadata
# 3. Collection switching
# 4. Cross-collection search
```

Schema additions:
```sql
-- Add to vector_database.py
CREATE TABLE collections (
    collection_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    created_at TIMESTAMP,
    document_count INTEGER,
    chunk_count INTEGER
);

-- Add collection_id to documents and chunks tables
```

Collection operations:
- `create_collection(name, description)`
- `list_collections()`
- `switch_collection(name)`
- `delete_collection(name)`
- `merge_collections(source, target)`
- `export_collection(name, path)`

### Task 7.3: Duplicate Detection
Create `src/deduplication.py`:

```python
# Intelligent duplicate detection:
# 1. Content hash comparison
# 2. Fuzzy matching for near-duplicates
# 3. Metadata-based detection
# 4. User confirmation for ambiguous cases
```

Deduplication strategies:
- Exact match: SHA-256 hash
- Near-duplicate: MinHash LSH
- Semantic: Embedding similarity > 0.95
- Metadata: Same source + date

### Task 7.4: Re-indexing Tools
Create `src/reindex.py`:

```python
# Re-indexing utilities:
# 1. Update embeddings with new model
# 2. Re-chunk with different parameters
# 3. Refresh corrupted indices
# 4. Optimize database
```

Operations:
```python
class ReindexTool:
    def reembed_collection(self, collection_id, new_model=None):
        # Re-generate all embeddings
        
    def rechunk_documents(self, chunk_size=512, overlap=128):
        # Re-chunk all documents
        
    def rebuild_indices(self):
        # Rebuild vector indices
        
    def vacuum_database(self):
        # Optimize SQLite storage
```

### Task 7.5: Corpus Analytics
Create `src/corpus_analytics.py`:

```python
# Corpus statistics and analysis:
# 1. Document distribution by type
# 2. Token count statistics  
# 3. Coverage analysis
# 4. Embedding space visualization
```

Analytics to generate:
- Total documents/chunks/tokens
- Average document length
- Document type distribution
- Ingestion timeline
- Most similar document pairs
- Corpus growth over time
- Dead documents (no retrievals)

### Task 7.6: CLI Commands for Corpus Management
Add to `main.py`:

```python
@click.command()
@click.argument('path')
@click.option('--pattern', default='**/*', help='File pattern')
@click.option('--collection', default='default', help='Target collection')
@click.option('--dry-run', is_flag=True, help='Preview without ingesting')
def ingest(path, pattern, collection, dry_run):
    """Ingest documents into corpus"""
    # Implementation

@click.command()
def corpus():
    """Manage document corpus"""
    # Subcommands: list, stats, dedupe, reindex
```

## Testing Requirements
Create `test_phase_7.py`:
1. Test bulk ingestion with sample documents
2. Test duplicate detection
3. Test collection management
4. Test re-indexing operations
5. Test corpus statistics generation

Create `sample_corpus/` directory with test documents:
```
sample_corpus/
├── pdfs/
│   ├── sample1.pdf
│   └── sample2.pdf
├── html/
│   └── page1.html
└── markdown/
    └── doc1.md
```

## Output Requirements
Create `handoff/phase_7_complete.json`:
```json
{
  "timestamp": "ISO-8601 timestamp",
  "phase": 7,
  "created_files": [
    "src/corpus_manager.py",
    "src/corpus_organizer.py",
    "src/deduplication.py",
    "src/reindex.py",
    "src/corpus_analytics.py",
    "test_phase_7.py"
  ],
  "corpus_features": {
    "bulk_ingestion": true,
    "parallel_processing": true,
    "collections": true,
    "deduplication": true,
    "reindexing": true,
    "analytics": true
  },
  "ingestion_stats": {
    "test_documents_ingested": 0,
    "test_chunks_created": 0,
    "test_time_seconds": 0.0
  },
  "collections": {
    "default": {
      "documents": 0,
      "chunks": 0,
      "size_mb": 0.0
    }
  },
  "cli_commands": [
    "python main.py ingest <path>",
    "python main.py corpus list",
    "python main.py corpus stats",
    "python main.py corpus dedupe"
  ]
}
```

## Performance Requirements
- Ingest 100 documents in <60 seconds
- Parallel processing with 4+ workers
- Memory usage <4GB during ingestion
- Checkpointing every 10 documents
- Resume from interruption

## Validation Checklist
- [ ] Can ingest directory of mixed documents
- [ ] Progress bar shows accurate progress
- [ ] Duplicates are detected and skipped
- [ ] Collections can be created and switched
- [ ] Statistics are accurate
- [ ] Re-indexing works without data loss
- [ ] CLI commands function correctly
- [ ] Handoff file created

Remember: Good corpus management is essential for RAG quality. The next phase will test the system comprehensively.