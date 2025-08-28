# System Architecture Overview

## Introduction
The Local RAG System is a comprehensive Retrieval-Augmented Generation platform that combines document ingestion, vector search, and language model generation into a unified, production-ready system for intelligent document querying and analysis.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACES                          │
├─────────────────────────────────────────────────────────────────────┤
│  CLI Interface (main.py)     │  Interactive Chat  │  API Endpoints  │
│  • Rich terminal output      │  • Multi-turn      │  • REST API     │
│  • Command organization      │  • Context aware   │  • JSON responses│ 
│  • Progress tracking         │  • Source citation │  • Async support │
└─────────────────────────────┬───────────────────┬───────────────────┘
                              │                   │
                              ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│  RAG Pipeline (rag_pipeline.py)                                    │
│  • Query orchestration       • Context assembly                    │
│  • Method selection          • Response generation                 │
│  • Multi-turn support        • Source attribution                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL & SEARCH LAYER                      │
├─────────────────────────────────────────────────────────────────────┤
│  Vector Search              │  Keyword Search    │  Hybrid Search   │
│  • Semantic similarity      │  • FTS5 full-text  │  • Score fusion  │
│  • sqlite-vec acceleration  │  • Boolean queries │  • Best of both  │
│  • <1ms performance         │  • Exact matching  │  • Adaptive      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       STORAGE & DATABASE LAYER                     │
├─────────────────────────────────────────────────────────────────────┤
│  Vector Database (vector_database.py)                              │
│  • SQLite + sqlite-vec extension                                   │
│  • Document, chunk, and embedding storage                          │
│  • Collection-based organization                                   │
│  • ACID transactions and referential integrity                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PROCESSING & MANAGEMENT LAYER                  │
├─────────────────────────────────────────────────────────────────────┤
│  Corpus Manager     │ Corpus Organizer │ Analytics        │ Tools   │
│  • Bulk ingestion   │ • Collections    │ • Statistics     │• Dedupe │
│  • Parallel proc.   │ • Metadata mgmt  │ • Quality assess │• Reindex│
│  • Checkpointing    │ • Import/export  │ • Similarity     │• Backup │
│  • Resume support   │ • Merge ops      │ • Reporting      │• Repair │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       CORE SERVICES LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│  Embedding Service          │  LLM Wrapper       │  Document Ingest │
│  • sentence-transformers    │  • GGUF models     │  • Multi-format  │
│  • Batch processing         │  • Metal accel     │  • Text extraction│
│  • MPS/CUDA acceleration    │  • Streaming       │  • Chunking      │
│  • Memory management        │  • Local inference │  • Metadata      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  File System               │  Models              │  Configuration   │
│  • Document corpus         │  • LLM models        │  • System config │
│  • Vector database         │  • Embedding models  │  • User settings │
│  • Logs and checkpoints    │  • Model cache       │  • CLI defaults  │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

### Document Ingestion Flow
```
Documents → Document Loaders → Text Extraction → Document Chunking → Embedding Generation → Vector Storage
    ↓              ↓                ↓               ↓                  ↓                ↓
PDF/HTML/MD    Format-specific   Clean text     Overlapping       sentence-       SQLite +
   Files        parsers          content        chunks            transformers     sqlite-vec
    |              |                |               |                  |                |
    ├─Metadata     ├─Error         ├─Content      ├─Position         ├─Batch          ├─Index
    │ extraction   │ handling      │ validation   │ tracking         │ processing     │ optimization
    └─Format       └─Fallback      └─Quality      └─Overlap          └─GPU accel     └─ACID safety
     detection      strategies      checks         management         & caching       & integrity
```

### Query Processing Flow  
```
User Query → Query Enhancement → Retrieval → Context Assembly → LLM Generation → Response + Sources
     ↓             ↓                ↓            ↓               ↓                ↓
   Raw text    • Embedding       Multiple     Ranking &      Local LLM      Formatted
   input       • Intent recog    methods:     deduplication  (Gemma 3)      response with
     |         • Query expand   - Vector       & filtering      |            citations
     ├─Clean   ├─History        - Keyword        |             ├─Metal       ├─Source
     │ & parse │ context        - Hybrid         ├─Token       │ acceleration│ attribution
     └─Validate└─Enhancement     retrieval      │ management   └─Streaming   └─Confidence
                                   |             └─Context      generation    scoring
                                   └─Collection   assembly
                                    filtering    & prompt
                                                building
```

## Component Interactions

### Core Service Dependencies
```
RAG Pipeline
    ├── Vector Database (storage & search)
    ├── Embedding Service (query & document embeddings)  
    ├── LLM Wrapper (response generation)
    └── Document Ingestion (content processing)

Corpus Manager  
    ├── Vector Database (storage)
    ├── Embedding Service (batch embedding)
    ├── Document Ingestion (file processing)
    └── Checkpoint System (resume capability)

Corpus Organizer
    ├── Vector Database (collection management)
    └── Analytics Integration (statistics)

Document Deduplicator
    ├── Vector Database (document access)
    ├── Embedding Service (semantic similarity)
    └── Analytics Integration (quality metrics)

Reindex Tools
    ├── Vector Database (maintenance operations)
    ├── Embedding Service (re-embedding)
    └── Backup System (safety operations)

Corpus Analytics
    ├── Vector Database (data analysis)
    ├── Embedding Service (similarity analysis)
    └── Statistical Libraries (clustering & analysis)
```

## Input/Output Specifications

### Document Ingestion Inputs
**Supported Formats**:
- **.txt**: Plain text files (UTF-8 encoding)
- **.pdf**: PDF documents (text extraction via PyMuPDF)
- **.html/.htm**: HTML documents (BeautifulSoup parsing)
- **.md/.markdown**: Markdown documents (native parsing)

**Input Requirements**:
- Files must be readable with appropriate permissions
- Minimum content length: ~50 characters
- Maximum individual file size: ~100MB (configurable)
- Directory structure: No specific requirements, supports deep nesting

**Processing Parameters**:
```python
ingestion_params = {
    'chunk_size': 512,           # Characters per chunk
    'chunk_overlap': 128,        # Overlap between chunks
    'batch_size': 32,            # Embedding batch size
    'max_workers': 4,            # Parallel processing workers
    'collection_id': 'default'   # Target collection
}
```

### Document Ingestion Outputs
**Database Records**:
```sql
-- Document record
{
    doc_id: "uuid-string",
    source_path: "/path/to/document.pdf", 
    metadata: {file_size, file_type, ingestion_timestamp, ...},
    content_hash: "sha256-hash"
}

-- Chunk records  
{
    chunk_id: "doc_uuid_chunk_0",
    doc_id: "doc_uuid", 
    content: "text content",
    chunk_index: 0,
    token_count: 95,
    metadata: {position, overlap_info, ...}
}

-- Embedding records
{
    chunk_id: "doc_uuid_chunk_0",
    embedding_vector: [0.1, 0.2, ...],  # 384-dimensional vector
    created_at: "2025-08-27T..."
}
```

### Query Processing Inputs
**Query Types**:
- **Simple Questions**: "What is machine learning?"
- **Complex Queries**: "How do transformers differ from RNNs in terms of parallelization?"
- **Multi-part Questions**: "Explain neural networks and give examples of applications"
- **Contextual Queries**: Follow-up questions in conversation

**Query Parameters**:
```python
query_params = {
    'question': "user query string",
    'collection_id': 'target_collection',
    'method': 'hybrid',          # vector|keyword|hybrid
    'k': 5,                      # Documents to retrieve
    'similarity_threshold': 0.7,  # Minimum similarity
    'max_context_tokens': 3000   # LLM context limit
}
```

### Query Processing Outputs
**Structured Response**:
```python
{
    'answer': 'Generated response text with citations and explanations',
    'sources': [
        {
            'chunk_id': 'doc1_chunk_0',
            'document_name': 'ml_introduction.pdf',
            'content': 'Relevant text snippet from source',
            'similarity': 0.87,
            'page_number': 15,      # If available
            'section': 'Introduction' # If available
        }
    ],
    'context': 'Combined context used for generation',
    'confidence': 0.85,          # Response confidence score
    'retrieval_stats': {
        'method': 'hybrid',
        'chunks_retrieved': 5,
        'retrieval_time_ms': 12.3,
        'total_candidates': 1247
    },
    'generation_stats': {
        'model': 'gemma-3-4b-it-q4_0',
        'tokens_generated': 150,
        'generation_time_ms': 890.5,
        'tokens_per_second': 27.5
    }
}
```

## Performance Characteristics

### System Performance Metrics
**End-to-End Performance**:
- **Query Response Time**: 500ms - 3s (typical)
- **Ingestion Throughput**: 5-20 documents/second
- **Search Performance**: <1ms vector search, <5ms keyword search
- **Memory Usage**: 2-8GB depending on models and corpus size
- **Storage Efficiency**: ~100-300MB per 10k documents

### Component Performance Breakdown
| Component | Operation | Performance | Bottleneck |
|-----------|-----------|-------------|------------|
| Document Ingestion | File processing | 5-20 docs/sec | I/O & parsing |
| Embedding Service | Text embedding | 100-200 texts/sec | Model inference |
| Vector Database | Similarity search | <1ms per query | sqlite-vec performance |
| LLM Wrapper | Text generation | 20-50 tokens/sec | Model size & hardware |
| Corpus Analytics | Statistics | <1s basic, 30s complex | Computation complexity |

### Scalability Characteristics
**Tested Limits**:
- **Document Count**: Up to 100k documents
- **Corpus Size**: Up to 10GB of text content
- **Concurrent Users**: 10-50 depending on hardware
- **Query Throughput**: 100-1000 queries/hour
- **Embedding Dimensions**: 128-2048 (optimal: 384-768)

## Technology Stack

### Core Technologies
**Database & Storage**:
- **SQLite**: Primary database engine for ACID compliance and simplicity
- **sqlite-vec**: Vector similarity search extension (critical performance component)
- **File System**: Direct file access for document corpus

**Machine Learning**:
- **sentence-transformers**: Embedding model framework
- **PyTorch**: Deep learning backend with MPS acceleration
- **NumPy/SciPy**: Numerical computing for similarity calculations
- **scikit-learn**: Clustering and statistical analysis

**Language Models**:  
- **llama.cpp**: GGUF model inference with Metal acceleration
- **Gemma-3-4b-it**: Primary LLM for response generation (4.5GB, Q4 quantization)
- **Streaming Support**: Real-time response generation

**Python Ecosystem**:
- **Click**: Command-line interface framework
- **Rich**: Terminal UI and progress visualization
- **asyncio**: Asynchronous processing for performance
- **tqdm**: Progress tracking across components

### Infrastructure Components
**Development Environment**:
- **Python 3.11**: Core runtime
- **Conda Environment**: Dependency management (rag_env)
- **macOS ARM64**: Primary platform (with Metal acceleration)

**Performance Optimization**:
- **Metal Performance Shaders**: GPU acceleration on Apple Silicon
- **Parallel Processing**: Multi-worker document processing
- **Memory Mapping**: Efficient large file handling
- **Connection Pooling**: Database connection optimization

## Data Architecture

### Database Schema
**Relational Structure**: Normalized schema with proper foreign keys
```sql
Collections (collection metadata)
    ↓ (1:N)
Documents (file metadata, content hashes)
    ↓ (1:N)
Chunks (text segments, position info)
    ↓ (1:1) 
Embeddings (vector representations)
    ↓ (1:1)
Vector Search Table (sqlite-vec optimized storage)
```

**Collection Support**: Multi-tenant architecture with collection isolation
- Each document belongs to exactly one collection
- Collections provide logical separation and organization
- Cross-collection search supported
- Collection-specific analytics and management

### File System Organization
```
local_rag_system/
├── src/                     # Core components
│   ├── document_ingestion.py   # Document loading and chunking
│   ├── embedding_service.py    # Vector embedding generation  
│   ├── vector_database.py      # Storage and search
│   ├── rag_pipeline.py         # Main RAG orchestration
│   ├── corpus_manager.py       # Bulk processing
│   ├── corpus_organizer.py     # Collection management
│   ├── deduplication.py        # Duplicate detection
│   ├── reindex.py              # Maintenance tools
│   └── corpus_analytics.py     # Analysis and reporting
├── data/                    # Runtime data
│   ├── rag_vectors.db          # Main vector database
│   ├── checkpoints/            # Processing checkpoints
│   ├── realistic_full/         # Converted corpus files
│   └── backups/                # Database backups
├── models/                  # AI models
│   ├── llm/gemma-3-4b-it-q4_0.gguf     # Language model
│   └── embeddings/all-MiniLM-L6-v2/    # Embedding model
├── tests/                   # Test suites
├── documentation/           # Component documentation
└── main.py                 # CLI entry point
```

## Component Integration Patterns

### Service Initialization Pattern
**Lazy Loading**: Services initialized on-demand for performance
```python
# Factory pattern for service creation
def create_rag_pipeline(db_path: str, embedding_path: str, llm_path: str):
    return RAGPipeline(
        db_path=db_path,
        embedding_model_path=embedding_path,
        llm_model_path=llm_path
    )

# Services initialized with configuration
pipeline = create_rag_pipeline(
    db_path=DEFAULT_DB_PATH,
    embedding_path=DEFAULT_EMBEDDING_PATH, 
    llm_path=DEFAULT_LLM_PATH
)
```

### Error Propagation Strategy
**Graceful Degradation**: System continues operation despite component failures
- **Component Isolation**: Individual component failures don't crash system
- **Fallback Mechanisms**: Alternative methods when primary components fail
- **Error Context**: Rich error information for debugging
- **Recovery Procedures**: Automatic recovery from transient failures

### Resource Management Pattern
**Efficient Resource Usage**: Smart resource allocation and cleanup
```python
# Context manager pattern for resource safety
with EmbeddingService(model_path) as embedding_service:
    embeddings = embedding_service.embed_texts(texts)
# Automatic cleanup and GPU memory release

# Connection pooling for database access
with vector_db.get_connection() as conn:
    cursor = conn.cursor()
    # Database operations
# Automatic connection return to pool
```

## Configuration Management

### System Configuration
**Centralized Configuration**: Single source of configuration truth
```python
# main.py configuration constants
DEFAULT_DB_PATH = "data/rag_vectors.db"
DEFAULT_LLM_PATH = "models/llm/gemma-3-4b-it-q4_0.gguf"
DEFAULT_EMBEDDING_PATH = "models/embeddings/all-MiniLM-L6-v2"
DEFAULT_MAX_WORKERS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_RETRIEVAL_K = 5
```

### Component Configuration
**Per-Component Settings**: Each component configurable independently
```python
# Document ingestion configuration
ingestion_config = {
    'chunk_size': 512,
    'chunk_overlap': 128,
    'supported_formats': ['.txt', '.pdf', '.md', '.html']
}

# Embedding service configuration  
embedding_config = {
    'model_path': 'models/embeddings/all-MiniLM-L6-v2',
    'batch_size': 32,
    'device': 'mps',  # Auto-detected
    'max_sequence_length': 256
}

# Vector database configuration
database_config = {
    'db_path': 'data/rag_vectors.db',
    'embedding_dimension': 384,
    'enable_sqlite_vec': True,
    'connection_timeout': 30
}
```

## Security and Safety

### Data Protection
**Security Measures**: Protection of sensitive data and system integrity
- **Input Validation**: Sanitize all user inputs and file paths
- **Path Traversal Protection**: Prevent access outside authorized directories
- **SQL Injection Prevention**: Parameterized queries throughout
- **Model Safety**: Validate model files before loading

### Access Control
**Permission Management**: Appropriate access controls
- **File Permissions**: Respect system file permissions
- **Directory Access**: Validate directory access before operations
- **Database Security**: Protect database from unauthorized access
- **Configuration Security**: Secure storage of sensitive configuration

### Backup and Recovery  
**Data Protection**: Comprehensive backup and recovery procedures
- **Automatic Backups**: Before destructive operations
- **Point-in-time Recovery**: Restore to specific timestamps
- **Incremental Backups**: Efficient backup of changes
- **Integrity Validation**: Verify backup completeness

## Deployment Architecture

### Single-Machine Deployment
**Current Configuration**: Optimized for single-machine operation
- **Local Models**: All AI models run locally for privacy and control
- **File-based Storage**: SQLite database for simplicity and reliability
- **Process-based Parallelism**: Multi-worker processing within single application
- **Resource Efficiency**: Optimized for desktop/server deployment

### Scaling Considerations
**Future Scaling Options**: Architecture supports scaling approaches
- **Horizontal Scaling**: Distribute processing across multiple machines
- **Database Scaling**: Migrate to distributed database systems
- **Model Serving**: Separate model inference servers
- **Microservices**: Component separation for independent scaling

### Performance Optimization
**System-Level Optimizations**: Architecture optimized for performance
- **Database Optimization**: sqlite-vec extension for vector operations
- **GPU Acceleration**: Metal/CUDA for embedding and LLM inference
- **Memory Management**: Efficient memory usage patterns
- **I/O Optimization**: Parallel file processing and batching

## Integration Points

### External System Integration
**API-Ready Architecture**: Components designed for external integration
- **RESTful Patterns**: Services follow REST principles for API exposure
- **JSON Communication**: Standardized data interchange format  
- **Async Support**: Asynchronous processing for web integration
- **Stateless Operations**: Most operations are stateless for scalability

### Development Integration
**Developer-Friendly**: Architecture supports development workflows
- **Component Testing**: Each component independently testable
- **Mock Support**: Easy mocking for unit testing
- **Development Tools**: Rich debugging and introspection capabilities
- **Documentation**: Comprehensive documentation for all components

## Future Architecture Enhancements

### Planned Improvements
**Architectural Evolution**: Roadmap for future enhancements
- **Web Dashboard**: Browser-based corpus management interface
- **Real-time Analytics**: Live monitoring and alerting
- **Advanced Visualization**: Interactive corpus exploration tools
- **Multi-modal Support**: Support for images, audio, and video content

### Scalability Roadmap
**Scaling Strategy**: Path to enterprise-scale deployment
- **Distributed Processing**: Kubernetes-based scaling
- **Cloud Integration**: Support for cloud storage and compute
- **Enterprise Features**: Authentication, authorization, audit trails
- **High Availability**: Redundancy and failover capabilities

## Best Practices

### System Administration
**Operational Excellence**: Guidelines for system operation
- **Regular Maintenance**: Scheduled maintenance windows
- **Performance Monitoring**: Track system metrics and trends
- **Capacity Planning**: Monitor growth and plan for scaling
- **Security Updates**: Regular updates of dependencies and models

### Development Practices
**Code Quality**: Maintain high code quality standards
- **Component Isolation**: Clean interfaces between components
- **Error Handling**: Comprehensive error handling throughout
- **Testing Coverage**: Extensive test coverage for all components
- **Documentation**: Keep documentation synchronized with code changes