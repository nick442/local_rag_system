# Local RAG System Documentation

## Overview
This directory contains comprehensive documentation for all components of the Local RAG System - a production-ready Retrieval-Augmented Generation platform for intelligent document querying and corpus management.

## System Architecture
**Start here**: [System Architecture](system_architecture.md) - Complete overview of how all components fit together, data flows, and integration patterns.

## Core RAG Components
These components form the foundation of the RAG system:

### 📄 [Document Ingestion](document_ingestion.md)
- **Purpose**: Load, parse, and chunk documents from multiple formats
- **Key Classes**: DocumentLoader, DocumentChunker, DocumentIngestionService
- **Inputs**: PDF, HTML, Markdown, TXT files
- **Outputs**: Structured document chunks ready for embedding

### 🧠 [Embedding Service](embedding_service.md)  
- **Purpose**: Generate vector embeddings with GPU acceleration
- **Key Classes**: EmbeddingService, EmbeddingBatch
- **Performance**: 100-200 texts/second on Apple Silicon
- **Features**: Batch processing, memory management, device optimization

### 🗄️ [Vector Database](vector_database.md)
- **Purpose**: High-performance vector storage and similarity search  
- **Key Technology**: SQLite + sqlite-vec extension
- **Performance**: <1ms similarity search for 10k+ documents
- **Features**: Hybrid search (vector + keyword), collection support

### 🔄 [RAG Pipeline](rag_pipeline.md)
- **Purpose**: Orchestrate complete RAG workflow
- **Key Classes**: RAGPipeline
- **Features**: Multi-method retrieval, context assembly, LLM integration
- **Usage**: Single queries and multi-turn conversations

## Phase 7 Corpus Management
Advanced corpus management capabilities for production deployment:

### 📦 [Corpus Manager](corpus_manager.md)
- **Purpose**: Bulk document processing with parallel execution
- **Key Features**: Parallel processing, checkpointing, duplicate detection
- **Performance**: 5-20 documents/second with 4 workers
- **Safety**: Resume capability, error recovery, progress tracking

### 🏗️ [Corpus Organizer](corpus_organizer.md)  
- **Purpose**: Collection management and document organization
- **Key Features**: Collection lifecycle, metadata management, import/export
- **Organization**: Logical document grouping with rich metadata
- **Operations**: Merge, split, tag, and organize collections

### 🔍 [Deduplication System](deduplication.md)
- **Purpose**: Multi-method duplicate detection and resolution
- **Detection Methods**: Exact, fuzzy, semantic, and metadata-based
- **Scalability**: Handles 10k+ documents efficiently
- **Algorithms**: SHA-256 hashing, MinHash LSH, embedding similarity

### 🔧 [Re-indexing Tools](reindex_tools.md)
- **Purpose**: Database maintenance and optimization
- **Operations**: Re-embedding, re-chunking, index rebuilding, vacuum
- **Safety**: Automatic backups, transaction safety, validation
- **Use Cases**: Model upgrades, performance optimization, maintenance

### 📊 [Corpus Analytics](corpus_analytics.md)
- **Purpose**: Comprehensive analysis, quality assessment, reporting
- **Capabilities**: Statistics, quality scoring, similarity analysis, clustering
- **Reporting**: JSON/CSV/HTML exports, visualization-ready data
- **Quality**: Automated quality assessment with recommendations

## User Interfaces

### 🖥️ [CLI Interface](cli_interface.md)
- **Purpose**: Command-line interface for all system operations
- **Framework**: Click + Rich for beautiful terminal experience
- **Organization**: Logical command groups (ingest, collection, analytics, maintenance)
- **Features**: Progress bars, colored output, comprehensive help

## Component Interaction Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │────│   RAG Pipeline  │────│ Vector Database │
│   (main.py)     │    │ (rag_pipeline)  │    │(vector_database)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │ Embedding       │              │
         │              │ Service         │              │
         │              │ (embedding_svc) │              │
         │              └─────────────────┘              │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Corpus Manager  │    │ Document        │    │ LLM Wrapper     │
│ (bulk ingestion)│    │ Ingestion       │    │ (llm_wrapper)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       
         ▼                       ▼                       
┌─────────────────┐    ┌─────────────────┐    
│ Corpus          │    │ Deduplication   │    
│ Organizer       │    │ System          │    
│ (collections)   │    │ (deduplication) │    
└─────────────────┘    └─────────────────┘    
         │                       │              
         ▼                       ▼              
┌─────────────────┐    ┌─────────────────┐    
│ Corpus          │    │ Reindex Tools   │    
│ Analytics       │    │ (reindex)       │    
│ (analytics)     │    │                 │    
└─────────────────┘    └─────────────────┘    
```

## Data Flow Examples

### Document Ingestion Flow
```
1. User: python main.py ingest directory docs/ --collection research
2. CLI: Validates parameters, creates CorpusManager
3. CorpusManager: Scans directory, detects file types
4. Document Ingestion: Loads and chunks each document
5. Embedding Service: Generates embeddings for chunks (batch processing)
6. Vector Database: Stores documents, chunks, embeddings with collection_id
7. CLI: Displays progress bars and final statistics
```

### Query Processing Flow
```
1. User: python main.py query "What is machine learning?" --collection research
2. CLI: Validates parameters, creates RAGPipeline
3. RAG Pipeline: Processes query, generates query embedding
4. Vector Database: Performs hybrid search (vector + keyword)
5. RAG Pipeline: Assembles context from retrieved chunks
6. LLM Wrapper: Generates response using assembled context
7. CLI: Displays formatted answer with source attribution
```

### Collection Management Flow
```
1. User: python main.py collection create "AI Papers" --description "Academic AI papers"
2. CLI: Creates CorpusOrganizer, validates collection parameters  
3. Corpus Organizer: Creates collection record with metadata
4. Vector Database: Updates schema if needed, creates collection entry
5. CLI: Confirms collection creation with assigned collection ID
```

## Performance Optimization Guidelines

### System-Level Optimization
**Hardware Utilization**: Maximize system resource usage
- **CPU**: Parallel processing with optimal worker count
- **GPU**: Metal/CUDA acceleration for embeddings and LLM
- **Memory**: Efficient memory usage with batch processing
- **Storage**: SSD recommended for database and model storage

### Component-Level Optimization
**Individual Component Tuning**: Optimize each component for specific use cases
- **Embedding Service**: Tune batch sizes for memory/speed balance
- **Vector Database**: Optimize sqlite-vec indices for query patterns
- **Corpus Manager**: Adjust worker count and checkpoint frequency
- **LLM Wrapper**: Configure generation parameters for speed vs quality

### Query Optimization
**Search Performance**: Optimize for typical query patterns
- **Collection Scoping**: Use specific collections to reduce search space
- **Method Selection**: Choose optimal retrieval method for query type
- **Caching**: Cache frequent queries and model outputs
- **Index Maintenance**: Regular index rebuilding for optimal performance

## Troubleshooting Guide

### Common Issues
**Frequent Problems and Solutions**:

#### "No supported files found"
- **Issue**: Ingestion finds no processable files
- **Solution**: Check file extensions match supported formats (.txt, .pdf, .md, .html)
- **Debug**: Use `--dry-run` to preview which files would be processed

#### "sqlite-vec extension not loaded"  
- **Issue**: Vector search falls back to slow manual search
- **Solution**: Verify sqlite-vec installation: `pip install sqlite-vec`
- **Impact**: 1000x slower search performance without extension

#### "Model loading failed"
- **Issue**: Cannot load embedding or LLM models
- **Solution**: Verify model paths exist, check model file integrity
- **Debug**: Use `python main.py status` to verify model availability

#### "Out of memory during processing"
- **Issue**: System runs out of memory during large batch processing
- **Solution**: Reduce `--batch-size` and `--max-workers` parameters
- **Monitoring**: Use system memory monitoring during large ingestions

### Performance Issues
**System Performance Problems**:

#### Slow ingestion speed
- **Diagnosis**: Check CPU usage, GPU utilization, and I/O wait
- **Solutions**: Adjust worker count, batch size, or hardware resources
- **Optimization**: Use SSD storage, increase RAM, enable GPU acceleration

#### Slow query responses  
- **Diagnosis**: Check database size, index status, and search method
- **Solutions**: Rebuild indices, use collection scoping, optimize retrieval method
- **Monitoring**: Track query response times and identify bottlenecks

## Quick Start Guide

### Basic Setup
```bash
# 1. Verify system status
python main.py status

# 2. Create your first collection
python main.py collection create "My Documents" --description "Personal document collection"

# 3. Ingest documents  
python main.py ingest directory /path/to/docs --collection my_documents

# 4. Test retrieval
python main.py query "summarize the main topics" --collection my_documents

# 5. Interactive exploration
python main.py chat --collection my_documents
```

### Production Deployment
```bash
# 1. High-performance ingestion
python main.py ingest directory large_corpus/ --max-workers 8 --batch-size 64 --collection production

# 2. Quality assessment
python main.py analytics quality --collection production

# 3. Duplicate cleanup  
python main.py maintenance dedupe --collection production

# 4. Performance optimization
python main.py maintenance reindex --operation rebuild

# 5. System validation
python main.py maintenance validate
```

This documentation provides complete coverage of all system components, their interactions, and operational procedures for both development and production use cases.