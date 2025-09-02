"""
Retrieval Module for RAG System
High-level interface for retrieving relevant document chunks.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import tiktoken

import numpy as np

from .vector_database import VectorDatabase
from .embedding_service import EmbeddingService


class RetrievalResult:
    """Container for a single retrieval result."""
    
    def __init__(self, chunk_id: str, content: str, score: float, 
                 metadata: Dict[str, Any], doc_id: str, chunk_index: int):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.metadata = metadata
        self.doc_id = doc_id
        self.chunk_index = chunk_index
    
    def __repr__(self):
        return f"RetrievalResult(chunk_id='{self.chunk_id}', score={self.score:.3f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'score': self.score,
            'metadata': self.metadata,
            'doc_id': self.doc_id,
            'chunk_index': self.chunk_index
        }


class Retriever:
    """High-level retrieval interface for the RAG system."""
    
    def __init__(self, vector_db: VectorDatabase, embedding_service: EmbeddingService, 
                 max_context_tokens: int = 6000, encoding_name: str = "cl100k_base"):
        """
        Initialize the retriever.
        
        Args:
            vector_db: Vector database instance
            embedding_service: Embedding service instance
            max_context_tokens: Maximum context length in tokens
            encoding_name: Tokenizer encoding name
        """
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.max_context_tokens = max_context_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.logger = logging.getLogger(__name__)
    
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
    
    def _vector_retrieve(self, query: str, k: int, collection_id: Optional[str] = None) -> List[RetrievalResult]:
        """Retrieve using vector similarity search."""
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search for similar chunks with optional collection filtering
        results = self.vector_db.search_similar(query_embedding, k, collection_id=collection_id)
        
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                content=data['content'],
                score=score,
                metadata=data['metadata'],
                doc_id=data['doc_id'],
                chunk_index=data['chunk_index']
            )
            for chunk_id, score, data in results
        ]
    
    def _keyword_retrieve(self, query: str, k: int, collection_id: Optional[str] = None) -> List[RetrievalResult]:
        """Retrieve using keyword search."""
        results = self.vector_db.keyword_search(query, k, collection_id=collection_id)
        
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                content=data['content'],
                score=score,
                metadata=data['metadata'],
                doc_id=data['doc_id'],
                chunk_index=data['chunk_index']
            )
            for chunk_id, score, data in results
        ]
    
    def _hybrid_retrieve(self, query: str, k: int, collection_id: Optional[str] = None, alpha: float = 0.7) -> List[RetrievalResult]:
        """Retrieve using hybrid vector + keyword search."""
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Perform hybrid search with optional collection filtering
        results = self.vector_db.hybrid_search(query_embedding, query, k, alpha, collection_id=collection_id)
        
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                content=data['content'],
                score=score,
                metadata=data['metadata'],
                doc_id=data['doc_id'],
                chunk_index=data['chunk_index']
            )
            for chunk_id, score, data in results
        ]
    
    def retrieve_with_context(self, query: str, k: int = 5, 
                            context_window: int = 1, 
                            method: str = "vector") -> List[RetrievalResult]:
        """
        Retrieve chunks with surrounding context chunks.
        
        Args:
            query: Search query string
            k: Number of base results to retrieve
            context_window: Number of chunks before/after to include
            method: Retrieval method
            
        Returns:
            List of RetrievalResult objects with expanded context
        """
        # Get base results
        base_results = self.retrieve(query, k, method)
        
        # Expand with context
        expanded_results = []
        seen_chunks = set()
        
        for result in base_results:
            # Get surrounding chunks
            doc_chunks = self.vector_db.get_document_chunks(result.doc_id)
            
            # Find the position of current chunk
            chunk_positions = {chunk['chunk_id']: idx for idx, chunk in enumerate(doc_chunks)}
            current_pos = chunk_positions.get(result.chunk_id, -1)
            
            if current_pos == -1:
                # Fallback: just add the original result
                if result.chunk_id not in seen_chunks:
                    expanded_results.append(result)
                    seen_chunks.add(result.chunk_id)
                continue
            
            # Add context chunks
            start_pos = max(0, current_pos - context_window)
            end_pos = min(len(doc_chunks), current_pos + context_window + 1)
            
            for pos in range(start_pos, end_pos):
                chunk_data = doc_chunks[pos]
                if chunk_data['chunk_id'] not in seen_chunks:
                    # Use original score for the matched chunk, lower for context
                    score = result.score if pos == current_pos else result.score * 0.5
                    
                    context_result = RetrievalResult(
                        chunk_id=chunk_data['chunk_id'],
                        content=chunk_data['content'],
                        score=score,
                        metadata=chunk_data['metadata'],
                        doc_id=result.doc_id,
                        chunk_index=chunk_data['chunk_index']
                    )
                    expanded_results.append(context_result)
                    seen_chunks.add(chunk_data['chunk_id'])
        
        # Sort by score and return
        expanded_results.sort(key=lambda x: x.score, reverse=True)
        return expanded_results
    
    def filter_by_metadata(self, results: List[RetrievalResult], 
                          filters: Dict[str, Any]) -> List[RetrievalResult]:
        """
        Filter retrieval results by metadata criteria.
        
        Args:
            results: List of RetrievalResult objects
            filters: Dictionary of metadata filters
            
        Returns:
            Filtered list of RetrievalResult objects
        """
        filtered_results = []
        
        for result in results:
            match = True
            for key, expected_value in filters.items():
                if key not in result.metadata:
                    match = False
                    break
                
                actual_value = result.metadata[key]
                
                # Handle different comparison types
                if isinstance(expected_value, dict) and '$in' in expected_value:
                    # List membership check
                    if actual_value not in expected_value['$in']:
                        match = False
                        break
                elif isinstance(expected_value, dict) and '$regex' in expected_value:
                    # Regex match (simple contains check)
                    import re
                    if not re.search(expected_value['$regex'], str(actual_value)):
                        match = False
                        break
                else:
                    # Exact match
                    if actual_value != expected_value:
                        match = False
                        break
            
            if match:
                filtered_results.append(result)
        
        return filtered_results
    
    def assemble_context(self, results: List[RetrievalResult], 
                        include_metadata: bool = True,
                        deduplicate: bool = True) -> str:
        """
        Assemble retrieval results into a formatted context string.
        
        Args:
            results: List of RetrievalResult objects
            include_metadata: Whether to include source metadata
            deduplicate: Whether to remove duplicate content
            
        Returns:
            Formatted context string ready for LLM prompt
        """
        if not results:
            return ""
        
        context_parts = []
        seen_content = set() if deduplicate else None
        current_tokens = 0
        
        for result in results:
            # Check for duplicates
            if deduplicate:
                content_hash = hash(result.content.strip())
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
            
            # Estimate token count
            content_tokens = len(self.encoding.encode(result.content))
            
            # Check if adding this chunk would exceed token limit
            if current_tokens + content_tokens > self.max_context_tokens:
                break
            
            # Format the chunk
            chunk_text = result.content.strip()
            
            if include_metadata:
                source_info = self._format_source_info(result)
                formatted_chunk = f"[{source_info}]\n{chunk_text}\n"
            else:
                formatted_chunk = f"{chunk_text}\n"
            
            context_parts.append(formatted_chunk)
            current_tokens += content_tokens
        
        return "\n".join(context_parts)
    
    def _format_source_info(self, result: RetrievalResult) -> str:
        """Format source information for a result."""
        metadata = result.metadata
        
        # Extract filename from source path
        source_path = metadata.get('source', metadata.get('filename', 'Unknown'))
        if source_path and source_path != 'Unknown':
            filename = Path(source_path).name
        else:
            filename = 'Unknown'
        
        # Add page/chunk information if available
        info_parts = [f"Source: {filename}"]
        
        if 'page_number' in metadata:
            info_parts.append(f"Page: {metadata['page_number']}")
        elif result.chunk_index is not None:
            info_parts.append(f"Chunk: {result.chunk_index + 1}")
        
        # Add score information
        info_parts.append(f"Score: {result.score:.3f}")
        
        return " | ".join(info_parts)
    
    def get_chunk_context(self, chunk_id: str, window_size: int = 2) -> Optional[str]:
        """
        Get the context around a specific chunk.
        
        Args:
            chunk_id: ID of the target chunk
            window_size: Number of chunks before and after to include
            
        Returns:
            Formatted context string or None if chunk not found
        """
        chunk_data = self.vector_db.get_chunk_by_id(chunk_id)
        if not chunk_data:
            return None
        
        doc_id = chunk_data['doc_id']
        doc_chunks = self.vector_db.get_document_chunks(doc_id)
        
        # Find position of target chunk
        target_pos = None
        for i, chunk in enumerate(doc_chunks):
            if chunk['chunk_id'] == chunk_id:
                target_pos = i
                break
        
        if target_pos is None:
            return chunk_data['content']
        
        # Get context window
        start_pos = max(0, target_pos - window_size)
        end_pos = min(len(doc_chunks), target_pos + window_size + 1)
        
        context_chunks = doc_chunks[start_pos:end_pos]
        
        # Format context
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            chunk_pos = start_pos + i
            marker = " >>> " if chunk_pos == target_pos else "     "
            context_parts.append(f"{marker}[Chunk {chunk_pos + 1}] {chunk['content']}")
        
        return "\n".join(context_parts)
    
    def similarity_search_around_chunk(self, chunk_id: str, k: int = 5) -> List[RetrievalResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: Reference chunk ID
            k: Number of similar chunks to return
            
        Returns:
            List of similar RetrievalResult objects
        """
        # Get the chunk content
        chunk_data = self.vector_db.get_chunk_by_id(chunk_id)
        if not chunk_data:
            return []
        
        # Use the chunk content as query
        return self.retrieve(chunk_data['content'], k + 1, method="vector")[1:]  # Skip self
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        db_stats = self.vector_db.get_database_stats()
        embedding_info = self.embedding_service.get_model_info()
        
        return {
            'database': db_stats,
            'embedding_service': embedding_info,
            'max_context_tokens': self.max_context_tokens,
            'encoding': self.encoding.name
        }


def create_retriever(db_path: str, embedding_model_path: str, 
                    embedding_dimension: int = 384, **kwargs) -> Retriever:
    """
    Factory function to create a complete Retriever instance.
    
    Args:
        db_path: Path to the vector database
        embedding_model_path: Path to the embedding model
        embedding_dimension: Dimension of embeddings
        **kwargs: Additional arguments for Retriever
        
    Returns:
        Configured Retriever instance
    """
    from .vector_database import create_vector_database
    from .embedding_service import create_embedding_service
    
    # Create embedding service first to derive true embedding dimension
    embedding_service = create_embedding_service(embedding_model_path)
    try:
        true_dim = embedding_service.get_embedding_dimension()
    except Exception:
        true_dim = embedding_dimension
    # Create vector DB with validated/derived dimension
    vector_db = create_vector_database(db_path, true_dim)
    
    # Create retriever
    return Retriever(vector_db, embedding_service, **kwargs)
