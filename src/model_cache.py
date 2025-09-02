"""
Model Cache Singleton
Thread-safe, lazy-loading cache for embedding and LLM models.

This avoids repeatedly loading large models across the application lifecycle.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import os
import logging


class ModelCache:
    """Thread-safe singleton cache for model instances.

    - Embedding models are keyed by (resolved_path, device).
    - LLM models are keyed by (resolved_path, selected init params).

    Lazy-loads models on first access using a provided loader callable.
    """

    _instance: Optional["ModelCache"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        # Internal caches. Keys are tuples; see get_* methods for exact shape.
        # Using generic tuple keys keeps annotations simple and readable.
        self._embedding_models: Dict[tuple, Any] = {}
        self._llm_models: Dict[tuple, Any] = {}
        self._locks: Dict[Any, threading.Lock] = {}

        # Cache statistics
        self._stats = {
            "embed_hits": 0,
            "embed_misses": 0,
            "llm_hits": 0,
            "llm_misses": 0,
        }

        # Configurable subset of LLM init params to use for cache key
        # Can be overridden via env var LLM_CACHE_PARAM_KEYS="k1,k2,k3"
        env_keys = os.environ.get("LLM_CACHE_PARAM_KEYS")
        self._llm_cache_param_keys: Optional[Tuple[str, ...]] = (
            tuple(k.strip() for k in env_keys.split(",") if k.strip()) if env_keys else None
        )

        self._logger = logging.getLogger(__name__)

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
        # Path validation for local filesystem paths with safe resolution
        path_obj = Path(model_path).expanduser()
        if path_obj.exists():
            try:
                resolved = str(path_obj.resolve(strict=True))
            except PermissionError as e:
                # Surface permission errors explicitly for local paths
                self._logger.error(
                    "Permission denied resolving embedding model path: %s", model_path
                )
                raise
            except OSError as e:
                # Unexpected resolution error – bubble up with context
                self._logger.error(
                    "Error resolving embedding model path '%s': %s", model_path, e
                )
                raise
        else:
            # Likely a model identifier (e.g., Hugging Face). Do not raise.
            # Keep original identifier as part of cache key and warn once.
            resolved = str(model_path)
            self._logger.debug(
                "Embedding model path does not exist locally; treating as identifier: %s",
                model_path,
            )

        key = (resolved, device or "auto")
        model = self._embedding_models.get(key)
        if model is not None:
            self._stats["embed_hits"] += 1
            return model

        lock = self._get_lock(("embed", key))
        with lock:
            # Double-check inside lock
            model = self._embedding_models.get(key)
            if model is not None:
                self._stats["embed_hits"] += 1
                return model
            model = loader()
            self._embedding_models[key] = model
            self._stats["embed_misses"] += 1
            self._logger.debug("Embedding cache miss; loaded and cached: %s | %s", key[0], key[1])
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
        # Safely resolve LLM model path (must be a local file)
        path_obj = Path(model_path).expanduser()
        try:
            resolved_path = str(path_obj.resolve(strict=True))
        except FileNotFoundError:
            self._logger.error("LLM model path not found: %s", model_path)
            raise
        except PermissionError as e:
            self._logger.error("Permission denied accessing LLM model path: %s", model_path)
            raise
        except OSError as e:
            self._logger.error("Error resolving LLM model path '%s': %s", model_path, e)
            raise
        # Only parameters that impact model construction should key the cache
        default_param_keys = (
            "n_ctx",
            "n_batch",
            "n_threads",
            "n_gpu_layers",
            "add_bos_token",
            "echo",
        )
        # Priority: explicit param -> instance config -> default list
        keys = cache_param_keys or self._llm_cache_param_keys or default_param_keys
        key_params = tuple(sorted((k, init_params.get(k)) for k in keys))
        key = (resolved_path, key_params)

        model = self._llm_models.get(key)
        if model is not None:
            self._stats["llm_hits"] += 1
            return model

        lock = self._get_lock(("llm", key))
        with lock:
            model = self._llm_models.get(key)
            if model is not None:
                self._stats["llm_hits"] += 1
                return model
            model = loader()
            self._llm_models[key] = model
            self._stats["llm_misses"] += 1
            self._logger.debug("LLM cache miss; loaded and cached: %s", key[0])
            return model

    # -------- Maintenance / Stats --------
    def clear(self) -> None:
        """Clear all cached models. Does not call any unload hooks."""
        self._embedding_models.clear()
        self._llm_models.clear()
        self._locks.clear()
        # Do not reset stats on clear; they are cumulative for observability

    def stats(self) -> Dict[str, Any]:
        return {
            "embedding_models": len(self._embedding_models),
            "llm_models": len(self._llm_models),
            "embed_hits": self._stats["embed_hits"],
            "embed_misses": self._stats["embed_misses"],
            "llm_hits": self._stats["llm_hits"],
            "llm_misses": self._stats["llm_misses"],
        }

    def evict(self, key: tuple) -> bool:
        """Evict a cached model by its key.

        The key must match the internal cache key:
        - Embeddings: (resolved_path_or_identifier: str, device: str)
        - LLM: (resolved_path: str, key_params: tuple)

        Returns True if an entry was removed from either cache.
        """
        removed = False

        # Attempt graceful resource cleanup if possible
        def _cleanup(obj: Any) -> None:
            for method_name in ("close", "shutdown", "unload", "release"):
                try:
                    method = getattr(obj, method_name, None)
                    if callable(method):
                        method()
                        self._logger.debug(
                            "Called cleanup method '%s' on cached model for key: %s",
                            method_name,
                            key,
                        )
                        return
                except Exception as e:
                    # Log and continue with eviction to avoid leaks
                    self._logger.warning(
                        "Error during cleanup '%s' for key %s: %s",
                        method_name,
                        key,
                        e,
                    )

        obj = self._embedding_models.get(key)
        if obj is not None:
            _cleanup(obj)
            del self._embedding_models[key]
            removed = True

        obj = self._llm_models.get(key)
        if obj is not None:
            _cleanup(obj)
            del self._llm_models[key]
            removed = True
        # Remove associated lock if present
        embed_lock_key = ("embed", key)
        llm_lock_key = ("llm", key)
        with self._instance_lock:
            self._locks.pop(embed_lock_key, None)
            self._locks.pop(llm_lock_key, None)
        if removed:
            self._logger.debug("Evicted cache entry for key: %s", key)
        return removed

    def set_llm_cache_param_keys(self, keys: Tuple[str, ...]) -> None:
        """Override the default set of LLM init params used for the cache key."""
        self._llm_cache_param_keys = keys

    def log_stats(self, logger: Optional[logging.Logger] = None) -> None:
        """Log cache statistics (hits/misses and sizes)."""
        lg = logger or self._logger
        s = self.stats()
        lg.info(
            "ModelCache stats | embed: %d models, %d hits/%d misses | llm: %d models, %d hits/%d misses",
            s["embedding_models"], s["embed_hits"], s["embed_misses"],
            s["llm_models"], s["llm_hits"], s["llm_misses"],
        )
