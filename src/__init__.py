"""
RAG System Package
Core components for document ingestion, embedding, and retrieval.
"""

from .document_ingestion import (
    Document,
    DocumentChunk,
    DocumentLoader,
    TextLoader,
    PDFLoader,
    HTMLLoader,
    MarkdownLoader,
    DocumentChunker,
    DocumentIngestionService
)

from .embedding_service import (
    EmbeddingService,
    EmbeddingBatch,
    create_embedding_service
)

from .vector_database import (
    VectorDatabase,
    create_vector_database,
    create_vector_index,
)

from .model_cache import (
    ModelCache,
)

from .retriever import (
    RetrievalResult,
    Retriever,
    create_retriever
)

__version__ = "1.0.0"
__all__ = [
    # Document ingestion
    "Document",
    "DocumentChunk", 
    "DocumentLoader",
    "TextLoader",
    "PDFLoader", 
    "HTMLLoader",
    "MarkdownLoader",
    "DocumentChunker",
    "DocumentIngestionService",
    
    # Embedding service
    "EmbeddingService",
    "EmbeddingBatch",
    "create_embedding_service",
    
    # Vector database
    "VectorDatabase",
    "create_vector_database",
    "create_vector_index",
    
    # Model cache
    "ModelCache",
    
    # Retrieval
    "RetrievalResult",
    "Retriever",
    "create_retriever"
]
