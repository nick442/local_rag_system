"""
Reranker Service for two-stage retrieval.

Provides an optional cross-encoder based reranker that can re-score and reorder
retrieval results. Defaults to a no-op identity reranker when not configured.

Activation options (non-breaking defaults):
- Set environment variable `RAG_RERANKER_MODEL` to a CrossEncoder model name, or
- Pass `model_name` explicitly when constructing `RerankerService`.

If the model cannot be loaded or is not provided, the reranker will operate in
identity mode and return the input order unchanged.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class RerankerConfig:
    model_name: Optional[str] = None
    top_k: Optional[int] = None  # If provided, truncate to top-k after rerank


class RerankerService:
    """Cross-encoder based reranker with safe, optional dependencies."""

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.logger = logging.getLogger(__name__)
        cfg = config or RerankerConfig()

        # Resolve model from env if not provided
        if not cfg.model_name:
            cfg.model_name = os.getenv("RAG_RERANKER_MODEL")

        self.config = cfg
        self._enabled = bool(self.config.model_name)
        self._model = None

        if self._enabled:
            try:
                # Lazy import to keep default environments light-weight
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(self.config.model_name)
                self.logger.info(
                    "Reranker model loaded: %s", self.config.model_name
                )
            except Exception as e:
                self.logger.warning(
                    "Reranker disabled (failed to load %s): %s",
                    self.config.model_name,
                    e,
                )
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._model is not None

    def rerank(self, query: str, results: List[Any]) -> List[Any]:
        """Rerank retrieval results for a given query.

        Args:
            query: User query text
            results: List of RetrievalResult-like objects with a `.content` attribute

        Returns:
            Possibly re-ordered list of results. If disabled, returns input unchanged.
        """
        # Fast path when disabled or trivial input
        if not self.enabled or not results:
            return results

        try:
            # Prepare pairs for CrossEncoder: (query, passage)
            pairs = [(query, r.content) for r in results]
            scores = self._model.predict(pairs)

            # Attach scores and sort descending
            ordered = [r for _, r in sorted(zip(scores, results), key=lambda t: t[0], reverse=True)]

            # Optional top-k truncation
            if self.config.top_k is not None and self.config.top_k > 0:
                ordered = ordered[: self.config.top_k]
            return ordered
        except Exception as e:
            self.logger.warning("Reranker error; returning original order: %s", e)
            return results


def create_reranker_service(model_name: Optional[str] = None, top_k: Optional[int] = None) -> RerankerService:
    return RerankerService(RerankerConfig(model_name=model_name, top_k=top_k))

