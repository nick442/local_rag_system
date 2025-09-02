from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np



class VectorIndexInterface(ABC):
    """Abstract interface for vector index implementations."""

    @abstractmethod
    def insert_document(self, doc_id: str, source_path: str, metadata: Dict[str, Any], collection_id: str = "default") -> bool:
        raise NotImplementedError

    @abstractmethod
    def insert_chunk(self, chunk: Any, embedding: np.ndarray, collection_id: str = "default") -> bool:
        raise NotImplementedError

    @abstractmethod
    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_id: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def keyword_search(self, query: str, k: int = 5, collection_id: Optional[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 5,
        alpha: float = 0.7,
        collection_id: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        raise NotImplementedError
