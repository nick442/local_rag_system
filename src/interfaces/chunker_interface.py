from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Any


class ChunkerInterface(ABC):
    """Abstract interface for document chunkers."""

    @abstractmethod
    def chunk_document(self, document: Any) -> List[Any]:
        """Split a document into chunks."""
        raise NotImplementedError
