"""
RAG System Package
Lightweight package initializer. Avoids importing heavy submodules at import time
to keep optional dependencies out of CI/mock contexts and speed up startup.
"""

__version__ = "1.0.0"

# Public API names are documented here for reference; import directly from
# submodules, e.g., `from src.vector_database import VectorDatabase`.
__all__ = [
    # Document ingestion
    "document_ingestion",
    # Embedding service
    "embedding_service",
    # Vector database
    "vector_database",
    # Model cache
    "model_cache",
    # Retrieval
    "retriever",
]
