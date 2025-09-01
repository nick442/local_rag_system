"""
Model Cache
===========

Thread-safe, LRU-style cache specialized for model-related results (LLM outputs,
tokenization, etc.). Provides safe path normalization, configurable parameter keys
for cache key construction, eviction and clear operations, and basic statistics
with periodic logging.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import time
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Tuple


# ---------- Type aliases (readability over complex annotations) ----------
CacheKey = Hashable
CacheValue = Any
ParamDict = Mapping[str, Any]


def _normalize_device(device: Optional[str]) -> str:
    return (device or "").strip().lower()


def _safe_resolve(path_like: os.PathLike[str] | str) -> str:
    """Resolve a path safely even if it may not exist.

    Falls back to absolute/expanded path if strict resolution fails.
    """
    try:
        return str(Path(path_like).expanduser().resolve(strict=True))
    except Exception:
        # Fallbacks: try non-strict resolve (Py>=3.12 guarantees non-strict by default)
        try:
            return str(Path(path_like).expanduser().resolve(strict=False))
        except Exception:
            # Final fallback: abspath after expanduser
            return os.path.abspath(os.fspath(Path(path_like).expanduser()))


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    clears: int = 0
    last_report_ts: float = 0.0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.total) if self.total else 0.0


class ModelCache:
    """LRU cache for model-related computations with configurable key parameters.

    Thread-safe operations; supports explicit evict and clear for memory pressure scenarios.
    """

    def __init__(
        self,
        max_items: int = 1024,
        llm_param_keys: Optional[Iterable[str]] = None,
        log_interval: int = 1000,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Args:
            max_items: Maximum number of entries to retain (LRU eviction beyond this).
            llm_param_keys: Parameter names used to influence LLM cache keys. If None,
                uses environment var `LLM_CACHE_PARAM_KEYS` (comma-separated) or a sensible default.
            log_interval: Log statistics every N operations (get/put/evict/clear).
            logger: Optional logger; if None, uses module logger.
        """
        self._store: "OrderedDict[CacheKey, CacheValue]" = OrderedDict()
        self._lock = Lock()
        self.max_items = max(1, int(max_items))
        self.log_interval = max(0, int(log_interval))
        self.logger = logger or logging.getLogger(__name__)
        self.stats = CacheStats(last_report_ts=time())

        # Configure which parameters affect cache keys
        env_keys = os.getenv("LLM_CACHE_PARAM_KEYS")
        if llm_param_keys is not None:
            self.llm_param_keys = tuple(str(k) for k in llm_param_keys)
        elif env_keys:
            self.llm_param_keys = tuple(k.strip() for k in env_keys.split(",") if k.strip())
        else:
            # Default keys commonly impacting LLM determinism/outputs
            self.llm_param_keys = (
                "temperature",
                "top_p",
                "max_tokens",
                "n_ctx",
                "n_gpu_layers",
                "stop",
                "seed",
                "repeat_penalty",
            )

    # ---------- Key construction ----------
    def build_model_key(
        self,
        model_path: os.PathLike[str] | str,
        device: Optional[str] = None,
        params: Optional[ParamDict] = None,
    ) -> CacheKey:
        """Construct a stable cache key for a model instance or result.

        - Uses safe path resolution (with fallbacks) to normalize model_path
        - Normalizes device string to lower-case
        - Filters provided params to only those configured in `llm_param_keys`
        - Produces a hashable tuple key
        """
        norm_path = _safe_resolve(model_path)
        norm_device = _normalize_device(device)
        params = params or {}

        # Keep only relevant parameters in a deterministic order
        filtered: Tuple[Tuple[str, Any], ...] = tuple(
            (k, params.get(k)) for k in self.llm_param_keys if k in params
        )
        return (norm_path, norm_device, filtered)

    # ---------- Core cache operations ----------
    def get(self, key: CacheKey, default: Optional[CacheValue] = None) -> Optional[CacheValue]:
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                # Move to end to mark as recently used
                self._store.move_to_end(key)
                self.stats.hits += 1
            else:
                self.stats.misses += 1
                value = default
            self._maybe_log_stats()
            return value

    def put(self, key: CacheKey, value: CacheValue) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            self.stats.puts += 1

            # Enforce LRU capacity
            while len(self._store) > self.max_items:
                evicted_key, _ = self._store.popitem(last=False)
                self.stats.evictions += 1
                self.logger.debug(f"ModelCache LRU evicted: {evicted_key}")

            self._maybe_log_stats()

    def evict(self, key: CacheKey) -> bool:
        """Remove a specific item from cache. Returns True if removed."""
        with self._lock:
            removed = key in self._store
            if removed:
                del self._store[key]
                self.stats.evictions += 1
            self._maybe_log_stats()
            return removed

    def clear_cache(self) -> int:
        """Clear all cache entries. Returns number of entries removed."""
        with self._lock:
            size = len(self._store)
            self._store.clear()
            self.stats.clears += 1
            self._maybe_log_stats(force=True)
            return size

    # ---------- Statistics ----------
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            total = self.stats.total
            return {
                "size": len(self._store),
                "capacity": self.max_items,
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "puts": self.stats.puts,
                "evictions": self.stats.evictions,
                "clears": self.stats.clears,
                "hit_rate": (self.stats.hits / total) if total else 0.0,
            }

    def _maybe_log_stats(self, force: bool = False) -> None:
        if self.log_interval <= 0:
            return
        total_ops = self.stats.total + self.stats.puts + self.stats.evictions + self.stats.clears
        if force or (total_ops % self.log_interval == 0):
            s = self.get_statistics()
            self.logger.info(
                "ModelCache stats - size=%d hit_rate=%.2f hits=%d misses=%d evictions=%d",
                s["size"], s["hit_rate"], s["hits"], s["misses"], s["evictions"],
            )


__all__ = [
    "ModelCache",
    "CacheKey",
    "CacheValue",
    "ParamDict",
]

