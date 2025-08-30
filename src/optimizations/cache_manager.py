"""
Multi-level caching system for the RAG pipeline.

Provides intelligent caching for:
1. Query results (LLM responses and retrieval results)  
2. Embedding computations (text → embedding vectors)
3. Retrieval operations (similarity search results)
4. Document processing (parsed content and chunks)
5. LLM generation (prompt → response caching)
"""

import hashlib
import pickle
import time
import logging
import threading
from collections import OrderedDict, defaultdict
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from functools import wraps
import psutil


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str, default=None):
        """Get item from cache, moving it to end (most recent)."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return default
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting LRU items if necessary."""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                # Remove least recently used item
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.hit_rate(),
                'size': len(self.cache),
                'maxsize': self.maxsize
            }


class CacheManager:
    """
    Multi-level caching system for RAG pipeline optimization.
    
    Target: Achieve 60% cache hit rate to reduce redundant computations
    """
    
    def __init__(self, max_memory_mb: int = 500):
        """
        Initialize cache manager.
        
        Args:
            max_memory_mb: Maximum memory to use for all caches combined
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.logger = logging.getLogger(__name__)
        
        # Multi-level caches with different sizes based on usage patterns
        self.caches = {
            'query': LRUCache(maxsize=200),        # Full query responses
            'embedding': LRUCache(maxsize=2000),    # Text embeddings (most frequently accessed)
            'retrieval': LRUCache(maxsize=500),     # Retrieval results
            'generation': LRUCache(maxsize=100),    # LLM generation results  
            'document': LRUCache(maxsize=100),      # Processed documents
            'chunk': LRUCache(maxsize=1000),        # Document chunks
        }
        
        # Memory tracking
        self.memory_usage = defaultdict(int)
        self.memory_lock = threading.RLock()
        
        # Cache statistics
        self.global_stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'memory_pressure_cleanups': 0
        }
        
        # Cache key prefixes for organization
        self.key_prefixes = {
            'query': 'q_',
            'embedding': 'e_',
            'retrieval': 'r_',
            'generation': 'g_', 
            'document': 'd_',
            'chunk': 'c_'
        }
        
        self.logger.info(f"CacheManager initialized - Max memory: {max_memory_mb}MB")
    
    def _generate_cache_key(self, cache_type: str, *args, **kwargs) -> str:
        """
        Generate a consistent cache key from arguments.
        
        Args:
            cache_type: Type of cache (query, embedding, etc.)
            *args, **kwargs: Arguments to hash
            
        Returns:
            Unique cache key string
        """
        try:
            # Create a tuple of all arguments for hashing
            key_data = (args, tuple(sorted(kwargs.items())))
            
            # Special handling for numpy arrays
            processed_data = []
            for item in key_data:
                if isinstance(item, np.ndarray):
                    # Hash array shape and first/last few elements for efficiency
                    array_signature = (item.shape, item.dtype, 
                                     tuple(item.flat[:5]), tuple(item.flat[-5:]))
                    processed_data.append(array_signature)
                else:
                    processed_data.append(item)
            
            # Generate hash
            serialized = pickle.dumps(processed_data, protocol=pickle.HIGHEST_PROTOCOL)
            hash_key = hashlib.sha256(serialized).hexdigest()[:16]  # 16 chars for efficiency
            
            prefix = self.key_prefixes.get(cache_type, 'x_')
            return f"{prefix}{hash_key}"
            
        except Exception as e:
            # Fallback for non-serializable data
            self.logger.warning(f"Cache key generation failed: {e}")
            fallback_key = hashlib.md5(str((cache_type, args, kwargs)).encode()).hexdigest()[:16]
            return f"{self.key_prefixes.get(cache_type, 'x_')}{fallback_key}"
    
    def _estimate_memory_usage(self, value: Any) -> int:
        """Estimate memory usage of a cached value in bytes."""
        try:
            if isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (str, bytes)):
                return len(value.encode('utf-8')) if isinstance(value, str) else len(value)
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_memory_usage(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_memory_usage(k) + self._estimate_memory_usage(v) 
                          for k, v in value.items())
            else:
                # Fallback: pickle size estimate
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Conservative estimate
            return 1024  # 1KB default
    
    def _check_memory_pressure(self) -> bool:
        """Check if we're under memory pressure."""
        with self.memory_lock:
            total_memory = sum(self.memory_usage.values())
            return total_memory > self.max_memory_bytes
    
    def _cleanup_memory_pressure(self) -> Dict[str, int]:
        """Clean up caches when under memory pressure."""
        self.logger.warning("Memory pressure detected - cleaning up caches")
        
        cleaned_items = {}
        
        # Clear caches in order of importance (least important first)
        cleanup_order = ['document', 'chunk', 'generation', 'retrieval', 'embedding', 'query']
        
        for cache_type in cleanup_order:
            if not self._check_memory_pressure():
                break
                
            cache = self.caches[cache_type]
            original_size = cache.size()
            
            # Clear 50% of the cache
            items_to_remove = original_size // 2
            with cache.lock:
                for _ in range(items_to_remove):
                    if cache.cache:
                        key, value = cache.cache.popitem(last=False)  # Remove LRU items
                        memory_freed = self.memory_usage.get(key, 0)
                        with self.memory_lock:
                            self.memory_usage[key] = 0
                        
            cleaned_items[cache_type] = original_size - cache.size()
        
        self.global_stats['memory_pressure_cleanups'] += 1
        
        with self.memory_lock:
            remaining_memory = sum(self.memory_usage.values())
        
        self.logger.info(f"Memory cleanup completed - {cleaned_items}, "
                        f"remaining: {remaining_memory / (1024*1024):.1f}MB")
        
        return cleaned_items
    
    def get(self, cache_type: str, key: str, default=None) -> Any:
        """
        Get item from specified cache.
        
        Args:
            cache_type: Type of cache to query
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        self.global_stats['total_operations'] += 1
        
        if cache_type not in self.caches:
            self.logger.warning(f"Unknown cache type: {cache_type}")
            self.global_stats['cache_misses'] += 1
            return default
        
        cache = self.caches[cache_type]
        value = cache.get(key, default)
        
        if value is not default:
            self.global_stats['cache_hits'] += 1
        else:
            self.global_stats['cache_misses'] += 1
        
        return value
    
    def put(self, cache_type: str, key: str, value: Any) -> bool:
        """
        Store item in specified cache.
        
        Args:
            cache_type: Type of cache to store in
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successfully cached
        """
        if cache_type not in self.caches:
            self.logger.warning(f"Unknown cache type: {cache_type}")
            return False
        
        try:
            # Estimate memory usage
            memory_size = self._estimate_memory_usage(value)
            
            # Check if this single item would exceed memory limit
            if memory_size > self.max_memory_bytes // 2:
                self.logger.warning(f"Item too large to cache: {memory_size / (1024*1024):.1f}MB")
                return False
            
            # Check memory pressure and cleanup if needed
            if self._check_memory_pressure():
                self._cleanup_memory_pressure()
            
            # Store in cache
            cache = self.caches[cache_type]
            cache.put(key, value)
            
            # Update memory tracking
            with self.memory_lock:
                self.memory_usage[key] = memory_size
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache item: {e}")
            return False
    
    def get_or_compute(self, cache_type: str, compute_func, *args, **kwargs) -> Any:
        """
        Get from cache or compute and cache the result.
        
        Args:
            cache_type: Type of cache to use
            compute_func: Function to compute value if not cached
            *args, **kwargs: Arguments for compute function and cache key
            
        Returns:
            Cached or computed value
        """
        # Generate cache key
        cache_key = self._generate_cache_key(cache_type, *args, **kwargs)
        
        # Try to get from cache first
        cached_value = self.get(cache_type, cache_key)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        start_time = time.time()
        try:
            computed_value = compute_func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            # Cache the result
            if self.put(cache_type, cache_key, computed_value):
                self.logger.debug(f"Cached {cache_type} result in {computation_time:.3f}s")
            
            return computed_value
            
        except Exception as e:
            self.logger.error(f"Computation failed for {cache_type}: {e}")
            raise
    
    def cached_embedding(self, texts: Union[str, List[str]], embedding_func) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Cache embedding computations.
        
        Args:
            texts: Text(s) to embed
            embedding_func: Function to generate embeddings
            
        Returns:
            Cached or computed embeddings
        """
        if isinstance(texts, str):
            # Single text embedding
            return self.get_or_compute('embedding', embedding_func, texts)
        else:
            # Batch embedding with per-text caching
            results = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache for each text
            for i, text in enumerate(texts):
                cache_key = self._generate_cache_key('embedding', text)
                cached_embedding = self.get('embedding', cache_key)
                
                if cached_embedding is not None:
                    results.append(cached_embedding)
                else:
                    results.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Compute embeddings for uncached texts
            if uncached_texts:
                new_embeddings = embedding_func(uncached_texts)
                
                # Cache new embeddings and update results
                for j, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                    cache_key = self._generate_cache_key('embedding', text)
                    self.put('embedding', cache_key, embedding)
                    results[uncached_indices[j]] = embedding
            
            return results
    
    def cached_query(self, query_text: str, retrieval_func, generation_func, 
                    retrieval_args=None, generation_args=None) -> Dict[str, Any]:
        """
        Cache full query pipeline (retrieval + generation).
        
        Args:
            query_text: Query text
            retrieval_func: Function for retrieval
            generation_func: Function for generation
            retrieval_args: Arguments for retrieval function
            generation_args: Arguments for generation function
            
        Returns:
            Cached or computed query result
        """
        retrieval_args = retrieval_args or {}
        generation_args = generation_args or {}
        
        # Generate cache key for full query
        query_cache_key = self._generate_cache_key('query', query_text, retrieval_args, generation_args)
        
        # Check full query cache first
        cached_result = self.get('query', query_cache_key)
        if cached_result is not None:
            return cached_result
        
        # Check retrieval cache
        retrieval_cache_key = self._generate_cache_key('retrieval', query_text, retrieval_args)
        cached_retrieval = self.get('retrieval', retrieval_cache_key)
        
        if cached_retrieval is not None:
            contexts = cached_retrieval
        else:
            # Perform retrieval and cache result
            contexts = retrieval_func(query_text, **retrieval_args)
            self.put('retrieval', retrieval_cache_key, contexts)
        
        # Generate response (this part is usually not cached separately due to context dependency)
        response = generation_func(query_text, contexts, **generation_args)
        
        # Cache full query result
        full_result = {
            'query': query_text,
            'contexts': contexts,
            'response': response,
            'timestamp': time.time()
        }
        
        self.put('query', query_cache_key, full_result)
        
        return full_result
    
    def cache_decorator(self, cache_type: str):
        """
        Decorator to automatically cache function results.
        
        Args:
            cache_type: Type of cache to use
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.get_or_compute(cache_type, func, *args, **kwargs)
            return wrapper
        return decorator
    
    def clear_cache(self, cache_type: Optional[str] = None) -> Dict[str, int]:
        """
        Clear specific cache or all caches.
        
        Args:
            cache_type: Specific cache to clear (or all if None)
            
        Returns:
            Number of items cleared per cache type
        """
        cleared_counts = {}
        
        if cache_type and cache_type in self.caches:
            cache = self.caches[cache_type]
            cleared_counts[cache_type] = cache.size()
            cache.clear()
        else:
            # Clear all caches
            for cache_name, cache in self.caches.items():
                cleared_counts[cache_name] = cache.size()
                cache.clear()
        
        # Reset memory tracking
        with self.memory_lock:
            if cache_type:
                # Clear memory tracking for specific cache type
                prefix = self.key_prefixes.get(cache_type, 'x_')
                keys_to_remove = [k for k in self.memory_usage.keys() if k.startswith(prefix)]
                for key in keys_to_remove:
                    del self.memory_usage[key]
            else:
                # Clear all memory tracking
                self.memory_usage.clear()
        
        total_cleared = sum(cleared_counts.values())
        self.logger.info(f"Cleared {total_cleared} items from cache(s): {cleared_counts}")
        
        return cleared_counts
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'global_stats': self.global_stats.copy(),
            'cache_stats': {},
            'memory_usage': {}
        }
        
        # Get per-cache statistics
        for cache_type, cache in self.caches.items():
            stats['cache_stats'][cache_type] = cache.stats()
        
        # Calculate overall hit rate
        total_operations = self.global_stats['total_operations']
        if total_operations > 0:
            stats['global_stats']['hit_rate'] = (
                self.global_stats['cache_hits'] / total_operations
            )
        else:
            stats['global_stats']['hit_rate'] = 0.0
        
        # Memory statistics
        with self.memory_lock:
            total_memory = sum(self.memory_usage.values())
            stats['memory_usage'] = {
                'total_mb': total_memory / (1024 * 1024),
                'max_mb': self.max_memory_bytes / (1024 * 1024),
                'utilization': total_memory / self.max_memory_bytes if self.max_memory_bytes > 0 else 0,
                'tracked_items': len(self.memory_usage)
            }
        
        return stats
    
    def optimize_cache_sizes(self) -> Dict[str, int]:
        """
        Automatically optimize cache sizes based on usage patterns.
        
        Returns:
            New cache sizes after optimization
        """
        stats = self.get_cache_statistics()
        new_sizes = {}
        
        for cache_type, cache_stats in stats['cache_stats'].items():
            current_size = cache_stats['maxsize']
            hit_rate = cache_stats['hit_rate']
            usage = cache_stats['size'] / current_size if current_size > 0 else 0
            
            # Adjust size based on hit rate and usage
            if hit_rate > 0.8 and usage > 0.9:
                # High hit rate and near capacity - increase size
                new_size = min(current_size * 2, current_size + 500)
            elif hit_rate < 0.3 or usage < 0.3:
                # Low hit rate or low usage - decrease size
                new_size = max(current_size // 2, 50)
            else:
                # Keep current size
                new_size = current_size
            
            new_sizes[cache_type] = new_size
            
            # Update cache with new size
            if new_size != current_size:
                self.caches[cache_type].maxsize = new_size
                self.logger.info(f"Optimized {cache_type} cache size: {current_size} → {new_size}")
        
        return new_sizes
    
    def cleanup(self) -> None:
        """Clean up cache manager resources."""
        self.clear_cache()  # Clear all caches
        
        self.logger.info("CacheManager cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Avoid errors during shutdown