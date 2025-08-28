"""
Embedding Service for RAG System
Handles embedding generation using SentenceTransformers.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Union, Optional
import gc

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .document_ingestion import DocumentChunk


class EmbeddingService:
    """Service for generating embeddings from text using SentenceTransformers."""
    
    def __init__(self, model_path: str, batch_size: int = 32, device: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            model_path: Path to the SentenceTransformers model
            batch_size: Number of documents to process in each batch
            device: Device to use ('cpu', 'mps', 'cuda'). If None, auto-detect.
        """
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.device = device or self._get_optimal_device()
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._load_model()
    
    def _get_optimal_device(self) -> str:
        """Determine the best device for embedding generation."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the SentenceTransformers model."""
        try:
            self.logger.info(f"Loading embedding model from: {self.model_path}")
            self.model = SentenceTransformer(str(self.model_path), device=self.device)
            
            # Get model info
            max_seq_length = getattr(self.model, 'max_seq_length', 256)
            embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            self.logger.info(f"Model loaded successfully:")
            self.logger.info(f"  Device: {self.device}")
            self.logger.info(f"  Max sequence length: {max_seq_length}")
            self.logger.info(f"  Embedding dimension: {embedding_dimension}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            List of normalized embedding vectors as numpy arrays
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        progress_bar = tqdm(
            range(0, len(texts), self.batch_size),
            desc="Generating embeddings",
            disable=not show_progress
        )
        
        try:
            for start_idx in progress_bar:
                end_idx = min(start_idx + self.batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Generate embeddings for batch
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,  # Normalize to unit vectors
                        batch_size=len(batch_texts)
                    )
                
                all_embeddings.extend(batch_embeddings)
                
                # Clear cache to manage memory
                if self.device != "cpu":
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                
                # Update progress
                progress_bar.set_postfix({
                    'batch': f"{start_idx//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}",
                    'memory': f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "N/A"
                })
        
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
        
        finally:
            # Cleanup
            gc.collect()
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[DocumentChunk], show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            show_progress: Whether to show progress bar
            
        Returns:
            List of normalized embedding vectors
        """
        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(texts, show_progress=show_progress)
    
    async def embed_texts_async(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """
        Asynchronously generate embeddings for texts.
        Useful for concurrent processing with other operations.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.embed_texts(texts, show_progress)
        )
    
    async def embed_chunks_async(self, chunks: List[DocumentChunk], show_progress: bool = True) -> List[np.ndarray]:
        """
        Asynchronously generate embeddings for document chunks.
        """
        texts = [chunk.content for chunk in chunks]
        return await self.embed_texts_async(texts, show_progress=show_progress)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2)
    
    def batch_similarity(self, query_embedding: np.ndarray, embeddings: List[np.ndarray]) -> np.ndarray:
        """Calculate cosine similarity between query embedding and a batch of embeddings."""
        embeddings_matrix = np.vstack(embeddings)
        return np.dot(embeddings_matrix, query_embedding)
    
    def find_most_similar(self, query_embedding: np.ndarray, embeddings: List[np.ndarray], k: int = 5) -> List[tuple]:
        """
        Find the k most similar embeddings to the query.
        
        Args:
            query_embedding: The query embedding vector
            embeddings: List of embedding vectors to search
            k: Number of most similar embeddings to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity (highest first)
        """
        if not embeddings:
            return []
        
        similarities = self.batch_similarity(query_embedding, embeddings)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]  # Sort descending
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.model:
            return {}
        
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'embedding_dimension': self.get_embedding_dimension(),
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown'),
            'model_name': getattr(self.model, '_model_name', 'Unknown'),
            'batch_size': self.batch_size
        }
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.device != "cpu":
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        gc.collect()
    
    def __del__(self):
        """Cleanup when service is destroyed."""
        self.clear_cache()


class EmbeddingBatch:
    """Helper class for managing embedding batches with metadata."""
    
    def __init__(self, texts: List[str], metadata: Optional[List[dict]] = None):
        """
        Initialize embedding batch.
        
        Args:
            texts: List of texts to embed
            metadata: Optional list of metadata dictionaries for each text
        """
        self.texts = texts
        self.metadata = metadata or [{}] * len(texts)
        self.embeddings = None
        
        if len(self.texts) != len(self.metadata):
            raise ValueError("Number of texts and metadata entries must match")
    
    def set_embeddings(self, embeddings: List[np.ndarray]):
        """Set the embeddings for this batch."""
        if len(embeddings) != len(self.texts):
            raise ValueError("Number of embeddings must match number of texts")
        self.embeddings = embeddings
    
    def get_items(self):
        """Get all items in the batch as (text, embedding, metadata) tuples."""
        if self.embeddings is None:
            raise ValueError("Embeddings have not been generated yet")
        
        return list(zip(self.texts, self.embeddings, self.metadata))
    
    def __len__(self):
        return len(self.texts)
    
    def __iter__(self):
        return iter(self.get_items())


def create_embedding_service(model_path: str, **kwargs) -> EmbeddingService:
    """
    Factory function to create an EmbeddingService instance.
    
    Args:
        model_path: Path to the SentenceTransformers model
        **kwargs: Additional arguments for EmbeddingService
        
    Returns:
        Configured EmbeddingService instance
    """
    return EmbeddingService(model_path, **kwargs)