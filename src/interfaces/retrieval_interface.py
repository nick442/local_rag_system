from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class RetrievalInterface(ABC):
    """Abstract interface for retriever implementations."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5,
        method: str = "vector",
        collection_id: Optional[str] = None,
    ) -> List[Any]:
        """Retrieve relevant contexts for a query."""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return implementation statistics/metadata."""
        raise NotImplementedError

