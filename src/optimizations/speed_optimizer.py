"""
Speed optimization techniques for the RAG system.

Focuses on:
1. Vectorized operations with numpy and batch processing
2. Asynchronous I/O for parallel document processing
3. Intelligent caching strategies for expensive computations
4. SQL query optimization with prepared statements
5. Parallel processing with ThreadPoolExecutor
"""

import asyncio
import numpy as np
import hashlib
import pickle
import sqlite3
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from pathlib import Path
import aiofiles


class SpeedOptimizer:
    """
    Manages speed optimizations for the RAG system.
    
    Target: Increase token throughput by 25% (140.53 → ~175 tokens/sec)
    """
    
    def __init__(self, max_workers: int = 4, cache_size_mb: int = 200):
        """
        Initialize speed optimizer.
        
        Args:
            max_workers: Maximum number of worker threads
            cache_size_mb: Maximum cache size in MB
        """
        self.max_workers = max_workers
        self.cache_size_mb = cache_size_mb
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        
        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # In-memory cache for expensive operations
        self.operation_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        
        # SQL prepared statements cache
        self.prepared_statements = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SpeedOptimizer initialized - Workers: {max_workers}, "
                        f"Cache: {cache_size_mb}MB")
    
    def vectorize_embeddings(self, texts: List[str], embedding_func: Callable,
                           batch_size: int = 32) -> np.ndarray:
        """
        Batch embedding computation with vectorized operations.
        
        Args:
            texts: List of texts to embed
            embedding_func: Function to generate embeddings
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        start_time = time.time()
        embeddings = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Generate cache key for batch
            batch_key = self._generate_cache_key(batch)
            
            # Check cache first
            cached_result = self._get_from_cache(f"embeddings_{batch_key}")
            if cached_result is not None:
                embeddings.extend(cached_result)
                continue
            
            # Generate embeddings for batch
            batch_embeddings = embedding_func(batch)
            
            # Convert to numpy array if not already
            if not isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = np.array(batch_embeddings)
            
            # Optimize data type for memory efficiency
            if batch_embeddings.dtype == np.float64:
                batch_embeddings = batch_embeddings.astype(np.float32)
            
            # Cache result
            self._store_in_cache(f"embeddings_{batch_key}", batch_embeddings.tolist())
            
            embeddings.extend(batch_embeddings.tolist())
        
        # Convert to numpy array with optimal dtype
        result = np.array(embeddings, dtype=np.float32)
        
        processing_time = time.time() - start_time
        self.logger.debug(f"Vectorized {len(texts)} embeddings in {processing_time:.3f}s "
                         f"({len(texts)/processing_time:.1f} texts/sec)")
        
        return result
    
    async def async_document_processing(self, document_paths: List[Path],
                                      process_func: Callable,
                                      max_concurrent: int = 10) -> List[Any]:
        """
        Process multiple documents asynchronously.
        
        Args:
            document_paths: List of document file paths
            process_func: Function to process each document
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_doc(doc_path: Path) -> Tuple[Path, Any]:
            async with semaphore:
                try:
                    # Run CPU-bound processing in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, process_func, doc_path
                    )
                    return doc_path, result
                except Exception as e:
                    self.logger.error(f"Error processing {doc_path}: {e}")
                    return doc_path, None
        
        # Create tasks for all documents
        tasks = [process_single_doc(path) for path in document_paths]
        
        # Execute with progress tracking
        start_time = time.time()
        results = []
        
        for coro in asyncio.as_completed(tasks):
            doc_path, result = await coro
            results.append(result)
            
            # Log progress
            if len(results) % 10 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed
                self.logger.info(f"Processed {len(results)}/{len(document_paths)} "
                               f"documents ({rate:.1f} docs/sec)")
        
        total_time = time.time() - start_time
        self.logger.info(f"Async processing completed: {len(document_paths)} docs "
                        f"in {total_time:.2f}s ({len(document_paths)/total_time:.1f} docs/sec)")
        
        return [r for r in results if r is not None]
    
    async def async_file_reader(self, file_paths: List[Path]) -> Dict[Path, str]:
        """
        Read multiple files asynchronously.
        
        Args:
            file_paths: List of file paths to read
            
        Returns:
            Dictionary mapping file paths to their contents
        """
        async def read_single_file(file_path: Path) -> Tuple[Path, str]:
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    return file_path, content
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                return file_path, ""
        
        tasks = [read_single_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks)
        
        return {path: content for path, content in results}
    
    def cache_expensive_operation(self, cache_key_prefix: str = "op"):
        """
        Decorator to cache expensive operations.
        
        Args:
            cache_key_prefix: Prefix for cache keys
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{cache_key_prefix}_{self._generate_cache_key((args, kwargs))}"
                
                # Check cache
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Cache result (if not too large)
                try:
                    result_size = len(pickle.dumps(result))
                    if result_size < self.cache_size_bytes // 10:  # Max 10% of cache per item
                        self._store_in_cache(cache_key, result)
                        
                    self.logger.debug(f"Cached {func.__name__} result "
                                    f"({result_size} bytes, {execution_time:.3f}s)")
                except Exception as e:
                    self.logger.warning(f"Failed to cache {func.__name__} result: {e}")
                
                return result
                
            return wrapper
        return decorator
    
    def optimize_batch_operations(self, items: List[Any], 
                                operation_func: Callable,
                                batch_size: int = 32,
                                parallel: bool = True) -> List[Any]:
        """
        Optimize batch operations with parallel processing.
        
        Args:
            items: List of items to process
            operation_func: Function to apply to each batch
            batch_size: Size of each batch
            parallel: Whether to use parallel processing
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        # Split into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        start_time = time.time()
        
        if parallel and len(batches) > 1:
            # Process batches in parallel
            futures = [self.executor.submit(operation_func, batch) for batch in batches]
            results = []
            
            for future in as_completed(futures):
                try:
                    batch_result = future.result()
                    if isinstance(batch_result, list):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
        else:
            # Process sequentially
            results = []
            for batch in batches:
                try:
                    batch_result = operation_func(batch)
                    if isinstance(batch_result, list):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)
                except Exception as e:
                    self.logger.error(f"Sequential batch processing error: {e}")
        
        processing_time = time.time() - start_time
        self.logger.debug(f"Batch processing: {len(items)} items in {processing_time:.3f}s "
                         f"({len(items)/processing_time:.1f} items/sec)")
        
        return results
    
    def optimize_sql_queries(self, db_path: str) -> Dict[str, Any]:
        """
        Optimize SQL queries with prepared statements and indices.
        
        Args:
            db_path: Path to SQLite database
            
        Returns:
            Dictionary of optimization statistics
        """
        optimizations_applied = []
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Enable query optimization pragmas
                optimizations = [
                    ("PRAGMA optimize", "Query optimizer statistics"),
                    ("PRAGMA analysis_limit=1000", "Analysis limit for statistics"),
                    ("PRAGMA cache_size=-64000", "64MB cache size"),
                    ("PRAGMA temp_store=MEMORY", "Memory-based temporary storage"),
                    ("PRAGMA journal_mode=WAL", "Write-ahead logging"),
                    ("PRAGMA synchronous=NORMAL", "Balanced durability/performance"),
                ]
                
                for pragma, description in optimizations:
                    try:
                        cursor.execute(pragma)
                        optimizations_applied.append((pragma, description))
                    except Exception as e:
                        self.logger.warning(f"Failed to apply {pragma}: {e}")
                
                # Create performance indices if they don't exist
                indices = [
                    ("idx_documents_collection", 
                     "CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id)"),
                    ("idx_chunks_document", 
                     "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(doc_id)"),
                    ("idx_chunks_collection", 
                     "CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection_id)"),
                    ("idx_embeddings_chunk", 
                     "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(chunk_id)"),
                ]
                
                for index_name, sql in indices:
                    try:
                        cursor.execute(sql)
                        optimizations_applied.append((index_name, "Performance index"))
                    except Exception as e:
                        self.logger.warning(f"Failed to create index {index_name}: {e}")
                
                # Analyze tables for query optimization
                cursor.execute("ANALYZE")
                optimizations_applied.append(("ANALYZE", "Table statistics update"))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"SQL optimization failed: {e}")
        
        self.logger.info(f"Applied {len(optimizations_applied)} SQL optimizations")
        return {'optimizations': optimizations_applied}
    
    def get_prepared_statement(self, conn: sqlite3.Connection, 
                             sql: str, key: str = None) -> sqlite3.Cursor:
        """
        Get or create a prepared statement.
        
        Args:
            conn: SQLite connection
            sql: SQL statement
            key: Cache key (defaults to SQL hash)
            
        Returns:
            Prepared cursor
        """
        if key is None:
            key = hashlib.md5(sql.encode()).hexdigest()
        
        if key not in self.prepared_statements:
            cursor = conn.cursor()
            cursor.execute(sql)
            self.prepared_statements[key] = cursor
            
        return self.prepared_statements[key]
    
    def parallel_similarity_search(self, query_embedding: np.ndarray,
                                 embeddings_chunks: List[Tuple[np.ndarray, Any]],
                                 top_k: int = 5,
                                 chunk_size: int = 1000) -> List[Tuple[float, Any]]:
        """
        Parallel similarity search for large embedding collections.
        
        Args:
            query_embedding: Query embedding vector
            embeddings_chunks: List of (embedding, metadata) tuples
            top_k: Number of top results to return
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            List of (similarity_score, metadata) tuples
        """
        if not embeddings_chunks:
            return []
        
        # Split embeddings into chunks for parallel processing
        chunks = [embeddings_chunks[i:i + chunk_size] 
                 for i in range(0, len(embeddings_chunks), chunk_size)]
        
        def compute_chunk_similarities(chunk):
            """Compute similarities for a chunk of embeddings."""
            similarities = []
            for embedding, metadata in chunk:
                # Compute cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((similarity, metadata))
            return similarities
        
        start_time = time.time()
        
        if len(chunks) > 1:
            # Process chunks in parallel
            futures = [self.executor.submit(compute_chunk_similarities, chunk) 
                      for chunk in chunks]
            
            all_similarities = []
            for future in as_completed(futures):
                try:
                    chunk_similarities = future.result()
                    all_similarities.extend(chunk_similarities)
                except Exception as e:
                    self.logger.error(f"Similarity computation error: {e}")
        else:
            # Process single chunk
            all_similarities = compute_chunk_similarities(embeddings_chunks)
        
        # Sort by similarity and return top k
        all_similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = all_similarities[:top_k]
        
        processing_time = time.time() - start_time
        self.logger.debug(f"Parallel similarity search: {len(embeddings_chunks)} embeddings "
                         f"in {processing_time:.3f}s")
        
        return top_results
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate a hash key for caching."""
        try:
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(serialized).hexdigest()
        except Exception:
            # Fallback for non-serializable data
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Any:
        """Get item from cache."""
        if key in self.operation_cache:
            self.cache_stats['hits'] += 1
            return self.operation_cache[key]
        
        self.cache_stats['misses'] += 1
        return None
    
    def _store_in_cache(self, key: str, value: Any) -> None:
        """Store item in cache with size management."""
        try:
            # Estimate size
            value_size = len(pickle.dumps(value))
            
            # Check if cache is getting too large
            if self.cache_stats['size'] + value_size > self.cache_size_bytes:
                self._cleanup_cache()
            
            # Store in cache
            self.operation_cache[key] = value
            self.cache_stats['size'] += value_size
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    def _cleanup_cache(self, target_reduction: float = 0.3) -> None:
        """Clean up cache when it gets too large."""
        if not self.operation_cache:
            return
        
        original_size = len(self.operation_cache)
        target_size = int(original_size * (1 - target_reduction))
        
        # Simple LRU-like cleanup (remove arbitrary items)
        # In a production system, you'd track access times
        keys_to_remove = list(self.operation_cache.keys())[:original_size - target_size]
        
        for key in keys_to_remove:
            del self.operation_cache[key]
        
        # Reset size counter (approximate)
        self.cache_stats['size'] = int(self.cache_stats['size'] * (1 - target_reduction))
        
        self.logger.info(f"Cache cleanup: {len(keys_to_remove)} items removed")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = 0
        if self.cache_stats['hits'] + self.cache_stats['misses'] > 0:
            cache_hit_rate = self.cache_stats['hits'] / (
                self.cache_stats['hits'] + self.cache_stats['misses']
            )
        
        return {
            'cache_stats': self.cache_stats.copy(),
            'cache_hit_rate': cache_hit_rate,
            'cache_items': len(self.operation_cache),
            'cache_size_mb': self.cache_stats['size'] / (1024 * 1024),
            'max_workers': self.max_workers,
            'prepared_statements': len(self.prepared_statements)
        }
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear all caches."""
        items_cleared = len(self.operation_cache)
        statements_cleared = len(self.prepared_statements)
        
        self.operation_cache.clear()
        self.prepared_statements.clear()
        self.cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        
        self.logger.info(f"Cache cleared: {items_cleared} items, {statements_cleared} statements")
        
        return {
            'operation_cache_items': items_cleared,
            'prepared_statements': statements_cleared
        }
    
    def shutdown(self) -> None:
        """Shutdown the optimizer and clean up resources."""
        self.executor.shutdown(wait=True)
        self.clear_cache()
        self.logger.info("SpeedOptimizer shutdown completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except:
            pass  # Avoid errors during shutdown