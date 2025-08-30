"""
Database performance optimization for SQLite with sqlite-vec extension.

Focuses on:
1. SQLite configuration optimization (WAL mode, cache settings, memory mapping)
2. Index creation and optimization for vector and text search
3. Connection pooling and prepared statements
4. Database maintenance (VACUUM, ANALYZE, integrity checks)
5. Query optimization and performance monitoring
"""

import sqlite3
import logging
import time
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import queue


class DatabaseOptimizer:
    """
    Optimizes SQLite database performance for the RAG system.
    
    Target: Minimize retrieval latency while maintaining data integrity
    """
    
    def __init__(self, db_path: str, max_connections: int = 10):
        """
        Initialize database optimizer.
        
        Args:
            db_path: Path to SQLite database file
            max_connections: Maximum number of pooled connections
        """
        self.db_path = Path(db_path)
        self.max_connections = max_connections
        self.logger = logging.getLogger(__name__)
        
        # Connection pool
        self._connection_pool = queue.Queue(maxsize=max_connections)
        self._pool_lock = threading.Lock()
        self._pool_initialized = False
        
        # Performance tracking
        self.performance_stats = {
            'queries_executed': 0,
            'total_query_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'connections_created': 0,
            'optimizations_applied': 0
        }
        
        # Prepared statements cache
        self.prepared_statements = {}
        
        self.logger.info(f"DatabaseOptimizer initialized for {db_path}")
    
    def _initialize_connection_pool(self) -> None:
        """Initialize the connection pool with optimized connections."""
        if self._pool_initialized:
            return
            
        with self._pool_lock:
            if self._pool_initialized:
                return
                
            for _ in range(self.max_connections):
                conn = self._create_optimized_connection()
                self._connection_pool.put(conn)
                
            self._pool_initialized = True
            self.logger.info(f"Connection pool initialized with {self.max_connections} connections")
    
    def _create_optimized_connection(self) -> sqlite3.Connection:
        """Create a SQLite connection with optimal settings."""
        try:
            # Create connection with optimal parameters
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,  # Allow sharing between threads
                isolation_level=None,  # Autocommit mode for better performance
                timeout=30.0  # 30 second timeout
            )
            
            # Apply performance optimizations
            self._apply_connection_optimizations(conn)
            
            self.performance_stats['connections_created'] += 1
            return conn
            
        except Exception as e:
            self.logger.error(f"Failed to create optimized connection: {e}")
            raise
    
    def _apply_connection_optimizations(self, conn: sqlite3.Connection) -> None:
        """Apply SQLite performance optimizations to a connection."""
        optimizations = [
            # Journal and synchronization
            ("PRAGMA journal_mode=WAL", "Write-ahead logging for better concurrency"),
            ("PRAGMA synchronous=NORMAL", "Balanced durability/performance"),
            ("PRAGMA wal_autocheckpoint=1000", "WAL checkpoint every 1000 pages"),
            
            # Memory and cache
            ("PRAGMA cache_size=-128000", "128MB cache size"), 
            ("PRAGMA temp_store=MEMORY", "Store temporary data in memory"),
            ("PRAGMA mmap_size=268435456", "256MB memory mapping"),
            
            # Query optimization
            ("PRAGMA optimize", "Enable query optimizer"),
            ("PRAGMA analysis_limit=1000", "Limit analysis for better performance"),
            
            # Foreign keys and integrity
            ("PRAGMA foreign_keys=ON", "Enable foreign key constraints"),
            ("PRAGMA case_sensitive_like=ON", "Case-sensitive LIKE for performance"),
            
            # Threading and locking
            ("PRAGMA busy_timeout=30000", "30 second busy timeout"),
        ]
        
        cursor = conn.cursor()
        
        for pragma, description in optimizations:
            try:
                cursor.execute(pragma)
                self.performance_stats['optimizations_applied'] += 1
                self.logger.debug(f"Applied optimization: {pragma}")
            except Exception as e:
                self.logger.warning(f"Failed to apply {pragma}: {e}")
        
        cursor.close()
    
    @contextmanager
    def get_connection(self):
        """
        Get an optimized database connection from the pool.
        
        Yields:
            sqlite3.Connection: Optimized database connection
        """
        if not self._pool_initialized:
            self._initialize_connection_pool()
        
        conn = None
        try:
            # Get connection from pool (blocking if pool is empty)
            conn = self._connection_pool.get(timeout=10)
            yield conn
        except queue.Empty:
            # Fallback: create temporary connection if pool exhausted
            self.logger.warning("Connection pool exhausted, creating temporary connection")
            temp_conn = self._create_optimized_connection()
            try:
                yield temp_conn
            finally:
                temp_conn.close()
        finally:
            # Return connection to pool
            if conn:
                try:
                    self._connection_pool.put_nowait(conn)
                except queue.Full:
                    # Pool is full, close excess connection
                    conn.close()
    
    def create_performance_indices(self) -> Dict[str, bool]:
        """
        Create optimized indices for better query performance.
        
        Returns:
            Dictionary indicating success/failure of each index creation
        """
        indices = {
            # Document table indices
            'idx_documents_collection': '''
                CREATE INDEX IF NOT EXISTS idx_documents_collection 
                ON documents(collection_id, created_at)
            ''',
            'idx_documents_path': '''
                CREATE INDEX IF NOT EXISTS idx_documents_path 
                ON documents(file_path)
            ''',
            
            # Chunks table indices
            'idx_chunks_document': '''
                CREATE INDEX IF NOT EXISTS idx_chunks_document 
                ON chunks(doc_id, chunk_index)
            ''',
            'idx_chunks_collection': '''
                CREATE INDEX IF NOT EXISTS idx_chunks_collection 
                ON chunks(collection_id, doc_id)
            ''',
            'idx_chunks_text_length': '''
                CREATE INDEX IF NOT EXISTS idx_chunks_text_length 
                ON chunks(LENGTH(text)) WHERE LENGTH(text) > 0
            ''',
            
            # Embeddings table indices  
            'idx_embeddings_chunk': '''
                CREATE INDEX IF NOT EXISTS idx_embeddings_chunk 
                ON embeddings(chunk_id)
            ''',
            
            # Collections table indices
            'idx_collections_name': '''
                CREATE INDEX IF NOT EXISTS idx_collections_name 
                ON collections(name)
            ''',
            
            # FTS indices for text search
            'idx_fts_chunks_collection': '''
                CREATE INDEX IF NOT EXISTS idx_fts_chunks_collection 
                ON chunks_fts(collection_id) WHERE collection_id IS NOT NULL
            ''',
        }
        
        results = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for index_name, sql in indices.items():
                try:
                    start_time = time.time()
                    cursor.execute(sql)
                    creation_time = time.time() - start_time
                    
                    results[index_name] = True
                    self.logger.info(f"Created index {index_name} in {creation_time:.3f}s")
                    
                except Exception as e:
                    results[index_name] = False
                    self.logger.error(f"Failed to create index {index_name}: {e}")
            
            # Commit all index creations
            conn.commit()
        
        successful_indices = sum(results.values())
        self.logger.info(f"Created {successful_indices}/{len(indices)} performance indices")
        
        return results
    
    def optimize_query_plans(self) -> Dict[str, Any]:
        """
        Optimize query execution plans by updating table statistics.
        
        Returns:
            Statistics about optimization results
        """
        start_time = time.time()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Update table statistics for query optimizer
                cursor.execute("ANALYZE")
                
                # Get table statistics
                cursor.execute("""
                    SELECT name, COUNT(*) as row_count 
                    FROM (
                        SELECT 'documents' as name UNION ALL
                        SELECT 'chunks' as name UNION ALL  
                        SELECT 'embeddings' as name UNION ALL
                        SELECT 'collections' as name
                    ) tables
                    LEFT JOIN (
                        SELECT 'documents' as table_name, COUNT(*) as count FROM documents UNION ALL
                        SELECT 'chunks', COUNT(*) FROM chunks UNION ALL
                        SELECT 'embeddings', COUNT(*) FROM embeddings UNION ALL
                        SELECT 'collections', COUNT(*) FROM collections
                    ) counts ON tables.name = counts.table_name
                    GROUP BY name
                """)
                
                table_stats = dict(cursor.fetchall())
                
                # Get index usage statistics
                cursor.execute("PRAGMA index_list(documents)")
                doc_indices = len(cursor.fetchall())
                
                cursor.execute("PRAGMA index_list(chunks)")  
                chunk_indices = len(cursor.fetchall())
                
                optimization_time = time.time() - start_time
                
                results = {
                    'optimization_time_seconds': optimization_time,
                    'table_statistics': table_stats,
                    'indices_count': {
                        'documents': doc_indices,
                        'chunks': chunk_indices
                    },
                    'analyze_completed': True
                }
                
                self.logger.info(f"Query plan optimization completed in {optimization_time:.3f}s")
                return results
                
            except Exception as e:
                self.logger.error(f"Query plan optimization failed: {e}")
                return {'analyze_completed': False, 'error': str(e)}
    
    def vacuum_and_optimize(self, full_vacuum: bool = False) -> Dict[str, Any]:
        """
        Perform database maintenance (VACUUM, ANALYZE).
        
        Args:
            full_vacuum: Whether to perform full VACUUM (slower but more thorough)
            
        Returns:
            Maintenance operation results
        """
        start_time = time.time()
        results = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Get initial database size
                cursor.execute("PRAGMA page_count")
                initial_pages = cursor.fetchone()[0]
                
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                
                initial_size_mb = (initial_pages * page_size) / (1024 * 1024)
                
                # Perform VACUUM operation
                if full_vacuum:
                    self.logger.info("Starting full VACUUM operation...")
                    cursor.execute("VACUUM")
                    results['vacuum_type'] = 'full'
                else:
                    # Incremental vacuum
                    cursor.execute("PRAGMA incremental_vacuum")
                    results['vacuum_type'] = 'incremental'
                
                # Update statistics
                cursor.execute("ANALYZE")
                
                # Get final database size
                cursor.execute("PRAGMA page_count")
                final_pages = cursor.fetchone()[0]
                final_size_mb = (final_pages * page_size) / (1024 * 1024)
                
                # Calculate space saved
                space_saved_mb = initial_size_mb - final_size_mb
                
                operation_time = time.time() - start_time
                
                results.update({
                    'initial_size_mb': initial_size_mb,
                    'final_size_mb': final_size_mb,
                    'space_saved_mb': space_saved_mb,
                    'operation_time_seconds': operation_time,
                    'pages_reclaimed': initial_pages - final_pages,
                    'success': True
                })
                
                self.logger.info(f"Database maintenance completed in {operation_time:.2f}s "
                               f"(Saved {space_saved_mb:.2f}MB)")
                
            except Exception as e:
                results.update({
                    'success': False,
                    'error': str(e),
                    'operation_time_seconds': time.time() - start_time
                })
                self.logger.error(f"Database maintenance failed: {e}")
        
        return results
    
    def optimize_fts_search(self) -> Dict[str, Any]:
        """
        Optimize Full-Text Search (FTS) configuration.
        
        Returns:
            FTS optimization results
        """
        optimizations = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Check if FTS table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='chunks_fts'
                """)
                
                if not cursor.fetchone():
                    self.logger.warning("FTS table 'chunks_fts' not found - skipping FTS optimization")
                    return {'fts_available': False}
                
                # Optimize FTS table
                cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
                optimizations['fts_optimize'] = True
                
                # Rebuild FTS index if needed
                cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
                optimizations['fts_rebuild'] = True
                
                # Get FTS statistics
                cursor.execute("SELECT COUNT(*) FROM chunks_fts")
                fts_doc_count = cursor.fetchone()[0]
                
                optimizations.update({
                    'fts_available': True,
                    'fts_document_count': fts_doc_count,
                    'success': True
                })
                
                self.logger.info(f"FTS optimization completed - {fts_doc_count} documents indexed")
                
            except Exception as e:
                optimizations.update({
                    'fts_available': True,
                    'success': False,
                    'error': str(e)
                })
                self.logger.error(f"FTS optimization failed: {e}")
        
        return optimizations
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database performance statistics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Database size and page information
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                
                cursor.execute("PRAGMA freelist_count")
                free_pages = cursor.fetchone()[0]
                
                database_size_mb = (page_count * page_size) / (1024 * 1024)
                free_space_mb = (free_pages * page_size) / (1024 * 1024)
                
                # Table row counts
                table_counts = {}
                tables = ['documents', 'chunks', 'embeddings', 'collections']
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        table_counts[table] = cursor.fetchone()[0]
                    except Exception as e:
                        self.logger.warning(f"Failed to count {table}: {e}")
                        table_counts[table] = 0
                
                # Index information
                cursor.execute("PRAGMA index_list(chunks)")
                chunk_indices = len(cursor.fetchall())
                
                cursor.execute("PRAGMA index_list(documents)")
                doc_indices = len(cursor.fetchall())
                
                # Compile statistics
                stats = {
                    'database_size_mb': database_size_mb,
                    'free_space_mb': free_space_mb,
                    'page_count': page_count,
                    'page_size': page_size,
                    'free_pages': free_pages,
                    'fragmentation_ratio': free_pages / page_count if page_count > 0 else 0,
                    'table_counts': table_counts,
                    'total_documents': table_counts.get('documents', 0),
                    'total_chunks': table_counts.get('chunks', 0),
                    'total_embeddings': table_counts.get('embeddings', 0),
                    'indices_count': {
                        'documents': doc_indices,
                        'chunks': chunk_indices
                    },
                    'performance_stats': self.performance_stats.copy()
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get database statistics: {e}")
                stats = {'error': str(e)}
        
        return stats
    
    def run_integrity_check(self) -> Dict[str, Any]:
        """
        Run database integrity check.
        
        Returns:
            Integrity check results
        """
        start_time = time.time()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Quick integrity check
                cursor.execute("PRAGMA quick_check")
                quick_check_results = cursor.fetchall()
                
                # Foreign key check
                cursor.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                
                check_time = time.time() - start_time
                
                is_ok = (
                    len(quick_check_results) == 1 and 
                    quick_check_results[0][0] == 'ok' and 
                    len(fk_violations) == 0
                )
                
                results = {
                    'integrity_ok': is_ok,
                    'quick_check_results': [row[0] for row in quick_check_results],
                    'foreign_key_violations': len(fk_violations),
                    'check_time_seconds': check_time,
                    'violations_details': fk_violations if fk_violations else []
                }
                
                if is_ok:
                    self.logger.info(f"Database integrity check passed in {check_time:.3f}s")
                else:
                    self.logger.warning(f"Database integrity issues detected: {results}")
                
                return results
                
            except Exception as e:
                return {
                    'integrity_ok': False,
                    'error': str(e),
                    'check_time_seconds': time.time() - start_time
                }
    
    def benchmark_query_performance(self, num_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark common query performance.
        
        Args:
            num_iterations: Number of iterations for each benchmark
            
        Returns:
            Performance benchmark results
        """
        benchmarks = {
            'document_count': "SELECT COUNT(*) FROM documents",
            'chunk_count': "SELECT COUNT(*) FROM chunks", 
            'recent_documents': "SELECT * FROM documents ORDER BY created_at DESC LIMIT 10",
            'chunks_by_document': """
                SELECT c.* FROM chunks c 
                INNER JOIN documents d ON c.doc_id = d.id 
                ORDER BY d.created_at DESC LIMIT 10
            """,
            'collection_stats': """
                SELECT collection_id, COUNT(*) as doc_count 
                FROM documents 
                GROUP BY collection_id
            """
        }
        
        results = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for benchmark_name, sql in benchmarks.items():
                times = []
                
                for _ in range(num_iterations):
                    try:
                        start_time = time.time()
                        cursor.execute(sql)
                        cursor.fetchall()  # Ensure all results are fetched
                        query_time = time.time() - start_time
                        times.append(query_time)
                        
                        self.performance_stats['queries_executed'] += 1
                        self.performance_stats['total_query_time'] += query_time
                        
                    except Exception as e:
                        self.logger.error(f"Benchmark query {benchmark_name} failed: {e}")
                        times.append(float('inf'))
                
                if times:
                    avg_time = sum(times) / len(times)
                    results[f'{benchmark_name}_avg_ms'] = avg_time * 1000
                    
        total_avg_time = (
            self.performance_stats['total_query_time'] / 
            max(self.performance_stats['queries_executed'], 1)
        )
        
        results['overall_avg_query_ms'] = total_avg_time * 1000
        
        self.logger.info(f"Query performance benchmark completed: "
                        f"avg {total_avg_time*1000:.2f}ms per query")
        
        return results
    
    def cleanup(self) -> None:
        """Clean up database optimizer resources."""
        # Close all connections in pool
        while not self._connection_pool.empty():
            try:
                conn = self._connection_pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
        
        # Clear prepared statements
        self.prepared_statements.clear()
        
        self.logger.info("DatabaseOptimizer cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Avoid errors during shutdown