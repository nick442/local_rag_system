"""
Model Cache Singleton
Thread-safe, lazy-loading cache for embedding and LLM models.

This avoids repeatedly loading large models across the application lifecycle.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


class ModelCache:
    """Thread-safe singleton cache for model instances.

    - Embedding models are keyed by (resolved_path, device).
    - LLM models are keyed by (resolved_path, selected init params).

    Lazy-loads models on first access using a provided loader callable.
    """

    _instance: Optional["ModelCache"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._embedding_models: Dict[Tuple[str, str], Any] = {}
        self._llm_models: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Any] = {}
        self._locks: Dict[Any, threading.Lock] = {}

    @classmethod
    def instance(cls) -> "ModelCache":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_lock(self, key: Any) -> threading.Lock:
        # One lock per key to prevent duplicate loads under concurrency
        with self._instance_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    # -------- Embedding Models --------
    def get_embedding_model(
        self,
        model_path: str,
        *,
        device: str = "auto",
        loader: Callable[[], Any],
    ) -> Any:
        """Get or load an embedding model.

        Args:
            model_path: Filesystem path or model identifier
            device: Device tag (e.g., 'cpu', 'mps', 'cuda', or 'auto')
            loader: Callable that returns a fully-initialized model instance

        Returns:
            A cached model instance
        """
        key = (str(Path(model_path).resolve()), device or "auto")
        model = self._embedding_models.get(key)
        if model is not None:
            return model

        lock = self._get_lock(("embed", key))
        with lock:
            # Double-check inside lock
            model = self._embedding_models.get(key)
            if model is not None:
                return model
            model = loader()
            self._embedding_models[key] = model
            return model

    # -------- LLM Models --------
    def get_llm_model(
        self,
        model_path: str,
        *,
        init_params: Dict[str, Any],
        loader: Callable[[], Any],
        cache_param_keys: Optional[Tuple[str, ...]] = None,
    ) -> Any:
        """Get or load an LLM model instance.

        Args:
            model_path: GGUF model path
            init_params: Full init parameters passed to the model
            loader: Callable that returns a fully-initialized model instance
            cache_param_keys: Subset of keys that affect model construction;
                              used to derive the cache key. If None, a default
                              subset is used.

        Returns:
            A cached model instance
        """
        resolved_path = str(Path(model_path).resolve())
        # Only parameters that impact model construction should key the cache
        default_param_keys = (
            "n_ctx",
            "n_batch",
            "n_threads",
            "n_gpu_layers",
            "add_bos_token",
            "echo",
        )
        keys = cache_param_keys or default_param_keys
        key_params = tuple(sorted((k, init_params.get(k)) for k in keys))
        key = (resolved_path, key_params)

        model = self._llm_models.get(key)
        if model is not None:
            return model

        lock = self._get_lock(("llm", key))
        with lock:
            model = self._llm_models.get(key)
            if model is not None:
                return model
            model = loader()
            self._llm_models[key] = model
            return model

    # -------- Maintenance / Stats --------
    def clear(self) -> None:
        """Clear all cached models. Does not call any unload hooks."""
        self._embedding_models.clear()
        self._llm_models.clear()
        self._locks.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "embedding_models": len(self._embedding_models),
            "llm_models": len(self._llm_models),
        }

