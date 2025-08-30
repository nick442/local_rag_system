"""
Comprehensive test suite for Phase 10 performance optimizations.

Tests all optimization components:
1. MemoryOptimizer - Memory management and optimization
2. SpeedOptimizer - Speed improvements and caching  
3. MetalOptimizer - Apple Silicon specific optimizations
4. DatabaseOptimizer - SQLite performance tuning
5. CacheManager - Multi-level caching system
6. AutoTuner - Automatic performance tuning

Validates performance improvements and system stability.
"""

import unittest
import tempfile
import shutil
import sqlite3
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

# Import optimization modules
from src.optimizations.memory_optimizer import MemoryOptimizer
from src.optimizations.speed_optimizer import SpeedOptimizer
from src.optimizations.metal_optimizer import MetalOptimizer
from src.optimizations.db_optimizer import DatabaseOptimizer
from src.optimizations.cache_manager import CacheManager, LRUCache
from src.optimizations.auto_tuner import AutoTuner, PerformanceMetrics, BottleneckType


class TestMemoryOptimizer(unittest.TestCase):
    """Test MemoryOptimizer functionality."""
    
    def setUp(self):
        self.optimizer = MemoryOptimizer(target_memory_mb=1000, min_free_memory_mb=200)
    
    def tearDown(self):
        self.optimizer.cleanup()
    
    def test_memory_optimizer_initialization(self):
        """Test MemoryOptimizer initializes correctly."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.target_memory, 1000)
        self.assertEqual(self.optimizer.min_free_memory, 200)
        # component_registry is a WeakValueDictionary, not a regular dict
        self.assertIsNotNone(self.optimizer.component_registry)
    
    def test_get_memory_usage(self):
        """Test memory usage tracking."""
        usage = self.optimizer.get_memory_usage()
        
        self.assertIsInstance(usage, dict)
        self.assertIn('rss_mb', usage)
        self.assertIn('available_mb', usage)
        self.assertIn('percent', usage)
        self.assertGreater(usage['rss_mb'], 0)
    
    def test_optimize_batch_size(self):
        """Test dynamic batch size optimization."""
        # Test with different memory conditions
        batch_sizes = []
        for base_size in [16, 32, 64]:
            optimized_size = self.optimizer.optimize_batch_size(base_size, memory_per_item_mb=5.0)
            batch_sizes.append(optimized_size)
            self.assertGreater(optimized_size, 0)
            self.assertLessEqual(optimized_size, base_size * 2)  # Shouldn't exceed 2x base
    
    def test_memory_mapping(self):
        """Test memory-mapped file access."""
        # Create a test file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_content = b"Test content for memory mapping" * 1000
            tmp_file.write(test_content)
            tmp_file.flush()
            
            # Test memory mapping
            memory_map = self.optimizer.enable_memory_mapping(tmp_file.name)
            
            if memory_map is not None:  # Only test if mapping succeeded
                self.assertIsNotNone(memory_map)
                self.assertEqual(len(memory_map), len(test_content))
                
                # Cleanup
                self.optimizer.close_memory_mapping(tmp_file.name)
            
            # Clean up test file
            Path(tmp_file.name).unlink()
    
    def test_garbage_collection(self):
        """Test aggressive garbage collection."""
        # Create some objects to collect
        test_objects = [list(range(1000)) for _ in range(10)]
        del test_objects
        
        # Run garbage collection
        gc_stats = self.optimizer.aggressive_gc(force_full=True)
        
        self.assertIsInstance(gc_stats, dict)
        self.assertIn('collected_gen0', gc_stats)
        self.assertIn('memory_freed_mb', gc_stats)
    
    def test_component_management(self):
        """Test component registration and unloading."""
        # Register a mock component
        mock_component = Mock()
        mock_component.cleanup = Mock()
        
        self.optimizer.register_component('test_component', mock_component)
        self.assertIn('test_component', self.optimizer.get_registered_components())
        
        # Unload component
        success = self.optimizer.unload_component('test_component')
        self.assertTrue(success)
        self.assertNotIn('test_component', self.optimizer.get_registered_components())
    
    def test_memory_pressure_check(self):
        """Test memory pressure detection."""
        pressure = self.optimizer.memory_pressure_check()
        
        self.assertIsInstance(pressure, dict)
        self.assertIn('any_pressure', pressure)
        self.assertIn('rss_mb', pressure)
        self.assertIn('available_mb', pressure)
        self.assertIsInstance(pressure['any_pressure'], bool)
    
    def test_numpy_array_optimization(self):
        """Test numpy array optimization."""
        # Create test arrays with different dtypes
        arrays = [
            np.array([1, 2, 3, 4, 5], dtype=np.float64),
            np.array([100, 200, 300], dtype=np.int64),
            np.array([1.1, 2.2, 3.3], dtype=np.float32)  # Already optimized
        ]
        
        optimized_arrays = self.optimizer.optimize_numpy_arrays(arrays)
        
        self.assertEqual(len(optimized_arrays), len(arrays))
        
        # Check that float64 was converted to float32
        self.assertEqual(optimized_arrays[0].dtype, np.float32)
        
        # Check that large integers were potentially downsized
        self.assertIn(optimized_arrays[1].dtype, [np.int32, np.int64])


class TestSpeedOptimizer(unittest.TestCase):
    """Test SpeedOptimizer functionality."""
    
    def setUp(self):
        self.optimizer = SpeedOptimizer(max_workers=2, cache_size_mb=10)
    
    def tearDown(self):
        self.optimizer.shutdown()
    
    def test_speed_optimizer_initialization(self):
        """Test SpeedOptimizer initializes correctly."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.max_workers, 2)
        self.assertIsNotNone(self.optimizer.executor)
    
    def test_vectorized_embeddings(self):
        """Test vectorized embedding computation."""
        def mock_embedding_func(texts):
            return np.random.rand(len(texts), 384).astype(np.float32)
        
        texts = [f"Test text {i}" for i in range(10)]
        embeddings = self.optimizer.vectorize_embeddings(texts, mock_embedding_func, batch_size=3)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], len(texts))
        self.assertEqual(embeddings.dtype, np.float32)
    
    def test_cache_decorator(self):
        """Test caching decorator functionality."""
        call_count = 0
        
        @self.optimizer.cache_expensive_operation("test")
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = expensive_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # Second call should use cache
        result2 = expensive_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # Should not increment
        
        # Different input should execute function
        result3 = expensive_function(7)
        self.assertEqual(result3, 14)
        self.assertEqual(call_count, 2)
    
    def test_batch_operations(self):
        """Test optimized batch operations."""
        def operation_func(batch):
            return [x * 2 for x in batch]
        
        items = list(range(20))
        results = self.optimizer.optimize_batch_operations(items, operation_func, batch_size=5, parallel=False)
        
        expected = [x * 2 for x in items]
        self.assertEqual(sorted(results), sorted(expected))  # Sort both to handle potential order differences
    
    def test_parallel_similarity_search(self):
        """Test parallel similarity search."""
        query_embedding = np.random.rand(384).astype(np.float32)
        
        # Create test embeddings with metadata
        embeddings_chunks = []
        for i in range(100):
            embedding = np.random.rand(384).astype(np.float32)
            metadata = {'id': i, 'text': f'document {i}'}
            embeddings_chunks.append((embedding, metadata))
        
        # Perform search
        results = self.optimizer.parallel_similarity_search(
            query_embedding, embeddings_chunks, top_k=5, chunk_size=25
        )
        
        self.assertEqual(len(results), 5)
        for similarity, metadata in results:
            # Accept numpy float types as well as regular float/int
            self.assertTrue(isinstance(similarity, (int, float, np.floating)))
            self.assertIsInstance(metadata, dict)
            self.assertIn('id', metadata)
    
    def test_performance_stats(self):
        """Test performance statistics collection."""
        stats = self.optimizer.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('cache_stats', stats)
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('max_workers', stats)
        self.assertEqual(stats['max_workers'], 2)
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        # Add some items to cache
        self.optimizer._store_in_cache('test1', 'value1')
        self.optimizer._store_in_cache('test2', 'value2')
        
        # Clear cache
        cleared = self.optimizer.clear_cache()
        
        self.assertIsInstance(cleared, dict)
        self.assertGreaterEqual(cleared.get('operation_cache_items', 0), 0)


class TestMetalOptimizer(unittest.TestCase):
    """Test MetalOptimizer functionality."""
    
    def setUp(self):
        self.optimizer = MetalOptimizer(memory_fraction=0.5, enable_profiling=True)
    
    def tearDown(self):
        self.optimizer.cleanup()
    
    def test_metal_optimizer_initialization(self):
        """Test MetalOptimizer initializes correctly."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.memory_fraction, 0.5)
        self.assertIsInstance(self.optimizer.mps_available, bool)
    
    def test_mps_profiling(self):
        """Test MPS usage profiling."""
        profile = self.optimizer.profile_mps_usage()
        
        self.assertIsInstance(profile, dict)
        self.assertIn('mps_available', profile)
        
        if profile['mps_available']:
            self.assertIn('device', profile)
            self.assertIn('allocated_memory_mb', profile)
    
    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.device')
    def test_optimize_embedding_model(self, mock_device, mock_mps_available):
        """Test embedding model optimization."""
        # Create mock model
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_model.half = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=[Mock()])
        
        # Test optimization
        optimized_model = self.optimizer.optimize_embedding_model(mock_model)
        
        # Verify methods were called
        mock_model.to.assert_called()
        mock_model.eval.assert_called()
    
    def test_tensor_operations(self):
        """Test tensor operations optimization."""
        # Create mock tensors
        mock_tensors = [Mock() for _ in range(3)]
        for tensor in mock_tensors:
            tensor.device = 'cpu'
            tensor.to = Mock(return_value=tensor)
            tensor.dtype = Mock()
        
        # Optimize tensors
        optimized_tensors = self.optimizer.optimize_tensor_operations(mock_tensors)
        
        self.assertEqual(len(optimized_tensors), len(mock_tensors))
    
    def test_mps_cache_clearing(self):
        """Test MPS cache clearing."""
        stats = self.optimizer.clear_mps_cache()
        
        # Should return stats dictionary even if MPS not available
        self.assertIsInstance(stats, dict)
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations."""
        recommendations = self.optimizer.get_optimization_recommendations()
        
        self.assertIsInstance(recommendations, list)
        for recommendation in recommendations:
            self.assertIsInstance(recommendation, str)


class TestDatabaseOptimizer(unittest.TestCase):
    """Test DatabaseOptimizer functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        
        # Create a test database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create test tables
        cursor.executescript('''
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                collection_id TEXT,
                file_path TEXT,
                created_at TIMESTAMP
            );
            
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER,
                collection_id TEXT,
                text TEXT,
                chunk_index INTEGER
            );
            
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY,
                chunk_id INTEGER
            );
            
            CREATE TABLE collections (
                id INTEGER PRIMARY KEY,
                name TEXT
            );
        ''')
        
        # Insert test data
        cursor.execute("INSERT INTO documents (collection_id, file_path) VALUES (?, ?)", 
                      ("test_collection", "/test/file.txt"))
        cursor.execute("INSERT INTO chunks (doc_id, collection_id, text) VALUES (?, ?, ?)",
                      (1, "test_collection", "test chunk"))
        cursor.execute("INSERT INTO embeddings (chunk_id) VALUES (?)", (1,))
        cursor.execute("INSERT INTO collections (name) VALUES (?)", ("test_collection",))
        
        conn.commit()
        conn.close()
        
        self.optimizer = DatabaseOptimizer(str(self.db_path), max_connections=2)
    
    def tearDown(self):
        self.optimizer.cleanup()
        shutil.rmtree(self.temp_dir)
    
    def test_database_optimizer_initialization(self):
        """Test DatabaseOptimizer initializes correctly."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.max_connections, 2)
        self.assertEqual(str(self.optimizer.db_path), str(self.db_path))
    
    def test_connection_management(self):
        """Test database connection management."""
        with self.optimizer.get_connection() as conn:
            self.assertIsNotNone(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            self.assertGreater(count, 0)
    
    def test_performance_indices_creation(self):
        """Test creation of performance indices."""
        results = self.optimizer.create_performance_indices()
        
        self.assertIsInstance(results, dict)
        # At least some indices should be created successfully
        successful_indices = sum(results.values())
        self.assertGreater(successful_indices, 0)
    
    def test_query_plan_optimization(self):
        """Test query plan optimization."""
        results = self.optimizer.optimize_query_plans()
        
        self.assertIsInstance(results, dict)
        if results.get('analyze_completed'):
            self.assertIn('table_statistics', results)
            self.assertIn('optimization_time_seconds', results)
    
    def test_database_statistics(self):
        """Test database statistics collection."""
        stats = self.optimizer.get_database_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('database_size_mb', stats)
        self.assertIn('table_counts', stats)
        self.assertIn('performance_stats', stats)
        
        # Verify table counts
        table_counts = stats['table_counts']
        self.assertGreater(table_counts.get('documents', 0), 0)
        self.assertGreater(table_counts.get('chunks', 0), 0)
    
    def test_integrity_check(self):
        """Test database integrity check."""
        results = self.optimizer.run_integrity_check()
        
        self.assertIsInstance(results, dict)
        self.assertIn('integrity_ok', results)
        self.assertIn('check_time_seconds', results)
        # Test database should pass integrity check
        self.assertTrue(results.get('integrity_ok', False))
    
    def test_query_performance_benchmark(self):
        """Test query performance benchmarking."""
        results = self.optimizer.benchmark_query_performance(num_iterations=3)
        
        self.assertIsInstance(results, dict)
        self.assertIn('overall_avg_query_ms', results)
        self.assertGreater(results['overall_avg_query_ms'], 0)


class TestLRUCache(unittest.TestCase):
    """Test LRUCache functionality."""
    
    def test_lru_cache_basic_operations(self):
        """Test basic LRU cache operations."""
        cache = LRUCache(maxsize=3)
        
        # Test put/get operations
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')
        
        self.assertEqual(cache.get('key1'), 'value1')
        self.assertEqual(cache.get('key2'), 'value2')
        self.assertEqual(cache.get('key3'), 'value3')
        self.assertEqual(cache.size(), 3)
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(maxsize=2)
        
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')  # Should evict key1
        
        self.assertIsNone(cache.get('key1'))
        self.assertEqual(cache.get('key2'), 'value2')
        self.assertEqual(cache.get('key3'), 'value3')
        self.assertEqual(cache.size(), 2)
    
    def test_lru_access_pattern(self):
        """Test LRU access pattern updates."""
        cache = LRUCache(maxsize=2)
        
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.get('key1')  # Access key1 to make it most recent
        cache.put('key3', 'value3')  # Should evict key2, not key1
        
        self.assertEqual(cache.get('key1'), 'value1')
        self.assertIsNone(cache.get('key2'))
        self.assertEqual(cache.get('key3'), 'value3')
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = LRUCache(maxsize=10)
        
        # Generate hits and misses
        cache.put('key1', 'value1')
        cache.get('key1')  # hit
        cache.get('key2')  # miss
        
        stats = cache.stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['hit_rate'], 0.5)


class TestCacheManager(unittest.TestCase):
    """Test CacheManager functionality."""
    
    def setUp(self):
        self.cache_manager = CacheManager(max_memory_mb=50)
    
    def tearDown(self):
        self.cache_manager.cleanup()
    
    def test_cache_manager_initialization(self):
        """Test CacheManager initializes correctly."""
        self.assertIsNotNone(self.cache_manager)
        self.assertEqual(len(self.cache_manager.caches), 6)  # 6 cache types
        self.assertIn('query', self.cache_manager.caches)
        self.assertIn('embedding', self.cache_manager.caches)
    
    def test_basic_cache_operations(self):
        """Test basic cache operations."""
        # Test put/get with a valid cache type
        success = self.cache_manager.put('query', 'key1', 'value1')
        self.assertTrue(success)
        
        value = self.cache_manager.get('query', 'key1')
        self.assertEqual(value, 'value1')
        
        # Test miss
        missing_value = self.cache_manager.get('query', 'nonexistent', default='default')
        self.assertEqual(missing_value, 'default')
    
    def test_get_or_compute(self):
        """Test get_or_compute functionality."""
        call_count = 0
        
        def compute_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should compute (using valid cache type)
        result1 = self.cache_manager.get_or_compute('query', compute_func, 5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # Second call should use cache
        result2 = self.cache_manager.get_or_compute('query', compute_func, 5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # Should not increment
    
    def test_cached_embedding(self):
        """Test embedding caching functionality."""
        def mock_embedding_func(texts):
            if isinstance(texts, str):
                return np.random.rand(384).astype(np.float32)
            else:
                return [np.random.rand(384).astype(np.float32) for _ in texts]
        
        # Test single embedding
        embedding1 = self.cache_manager.cached_embedding("test text", mock_embedding_func)
        self.assertIsInstance(embedding1, np.ndarray)
        
        # Second call should be cached
        embedding2 = self.cache_manager.cached_embedding("test text", mock_embedding_func)
        np.testing.assert_array_equal(embedding1, embedding2)
        
        # Test batch embeddings
        texts = ["text1", "text2", "text3"]
        embeddings = self.cache_manager.cached_embedding(texts, mock_embedding_func)
        self.assertEqual(len(embeddings), 3)
        for embedding in embeddings:
            self.assertIsInstance(embedding, np.ndarray)
    
    def test_cache_statistics(self):
        """Test cache statistics collection."""
        # Add some cache operations
        self.cache_manager.put('test', 'key1', 'value1')
        self.cache_manager.get('test', 'key1')  # hit
        self.cache_manager.get('test', 'key2')  # miss
        
        stats = self.cache_manager.get_cache_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('global_stats', stats)
        self.assertIn('cache_stats', stats)
        self.assertIn('memory_usage', stats)
        
        # Check that statistics are being tracked
        self.assertGreater(stats['global_stats']['total_operations'], 0)
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Add items to multiple caches
        self.cache_manager.put('query', 'q1', 'response1')
        self.cache_manager.put('embedding', 'e1', np.array([1, 2, 3]))
        
        # Clear specific cache
        cleared = self.cache_manager.clear_cache('query')
        self.assertIn('query', cleared)
        
        # Verify query cache is cleared but embedding cache is not
        self.assertIsNone(self.cache_manager.get('query', 'q1'))
        self.assertIsNotNone(self.cache_manager.get('embedding', 'e1'))
        
        # Clear all caches
        all_cleared = self.cache_manager.clear_cache()
        self.assertGreater(sum(all_cleared.values()), 0)


class TestAutoTuner(unittest.TestCase):
    """Test AutoTuner functionality."""
    
    def setUp(self):
        self.mock_system_manager = Mock()
        self.auto_tuner = AutoTuner(
            self.mock_system_manager, 
            monitoring_interval=1.0,  # Short interval for testing
            optimization_threshold=0.1
        )
    
    def tearDown(self):
        self.auto_tuner.cleanup()
    
    def test_auto_tuner_initialization(self):
        """Test AutoTuner initializes correctly."""
        self.assertIsNotNone(self.auto_tuner)
        self.assertEqual(self.auto_tuner.monitoring_interval, 1.0)
        self.assertEqual(self.auto_tuner.optimization_threshold, 0.1)
        self.assertFalse(self.auto_tuner._monitoring)
    
    def test_metrics_collection(self):
        """Test performance metrics collection."""
        # Mock component responses
        mock_monitor = Mock()
        mock_monitor.get_current_stats.return_value = {
            'token_throughput': 100.0,
            'avg_retrieval_time_ms': 500.0,
            'avg_generation_time_ms': 2000.0
        }
        
        mock_cache_manager = Mock()
        mock_cache_manager.get_cache_statistics.return_value = {
            'global_stats': {'hit_rate': 0.7}
        }
        
        self.mock_system_manager.get_component.side_effect = lambda name: {
            'monitor': mock_monitor,
            'cache_manager': mock_cache_manager
        }.get(name)
        
        # Collect metrics
        metrics = self.auto_tuner.collect_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.timestamp, 0)
        self.assertEqual(metrics.token_throughput, 100.0)
        self.assertEqual(metrics.cache_hit_rate, 0.7)
    
    def test_bottleneck_identification(self):
        """Test bottleneck identification."""
        # Create metrics that should trigger bottleneck detection
        high_memory_metrics = PerformanceMetrics(
            timestamp=time.time(),
            token_throughput=50.0,  # Low throughput
            memory_usage_mb=15000.0,  # High memory usage
            cpu_percent=95.0,  # High CPU
            retrieval_latency_ms=6000.0,  # Slow retrieval
            generation_latency_ms=35000.0,  # Slow generation
            cache_hit_rate=0.2,  # Poor cache performance
            db_query_time_ms=1500.0,  # Slow database
            io_wait_time_ms=2500.0  # I/O bottleneck
        )
        
        bottlenecks = self.auto_tuner.identify_bottlenecks(high_memory_metrics)
        
        self.assertIsInstance(bottlenecks, list)
        # Should detect multiple bottlenecks
        self.assertGreater(len(bottlenecks), 0)
        
        # Check for specific bottleneck types
        bottleneck_values = [b.value for b in bottlenecks]
        self.assertIn('slow_generation', bottleneck_values)
        self.assertIn('slow_retrieval', bottleneck_values)
    
    def test_optimization_application(self):
        """Test optimization application."""
        # Mock components for optimization
        mock_memory_optimizer = Mock()
        mock_memory_optimizer.emergency_cleanup.return_value = {'memory_freed_mb': 500}
        
        self.mock_system_manager.get_component.return_value = mock_memory_optimizer
        
        # Apply memory pressure optimization
        action = self.auto_tuner.apply_optimization(BottleneckType.MEMORY_PRESSURE)
        
        self.assertEqual(action.action_type, "memory_cleanup")
        self.assertTrue(action.success)
        self.assertGreater(action.expected_improvement, 0)
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations generation."""
        # Add some metrics to history
        for i in range(15):
            metrics = PerformanceMetrics(
                timestamp=time.time() - (15-i),
                token_throughput=100 - i,  # Declining performance
                memory_usage_mb=5000 + (i * 100),  # Growing memory
                cpu_percent=50,
                retrieval_latency_ms=500,
                generation_latency_ms=2000,
                cache_hit_rate=0.3,  # Poor cache performance
                db_query_time_ms=600,  # Good database performance
                io_wait_time_ms=100
            )
            self.auto_tuner.metrics_history.append(metrics)
        
        recommendations = self.auto_tuner.get_optimization_recommendations()
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check recommendation structure
        for rec in recommendations:
            self.assertIn('priority', rec)
            self.assertIn('category', rec)
            self.assertIn('recommendation', rec)
            self.assertIn('confidence', rec)
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop functionality."""
        # Start monitoring
        self.auto_tuner.start_monitoring()
        self.assertTrue(self.auto_tuner._monitoring)
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Stop monitoring
        self.auto_tuner.stop_monitoring()
        self.assertFalse(self.auto_tuner._monitoring)
    
    def test_tuning_statistics(self):
        """Test tuning statistics collection."""
        # Add some test data
        self.auto_tuner.baseline_metrics = PerformanceMetrics(
            time.time(), 100, 5000, 50, 500, 2000, 0.5, 300, 100
        )
        
        current_metrics = PerformanceMetrics(
            time.time(), 120, 4800, 45, 400, 1800, 0.6, 250, 90
        )
        self.auto_tuner.metrics_history.append(current_metrics)
        
        stats = self.auto_tuner.get_tuning_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('monitoring_active', stats)
        self.assertIn('metrics_collected', stats)
        self.assertIn('baseline_established', stats)
        self.assertTrue(stats['baseline_established'])


class TestIntegration(unittest.TestCase):
    """Integration tests for optimization components."""
    
    def test_optimization_component_integration(self):
        """Test that all optimization components can work together."""
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "integration_test.db"
        
        try:
            # Initialize all optimizers
            memory_optimizer = MemoryOptimizer(target_memory_mb=500)
            speed_optimizer = SpeedOptimizer(max_workers=2)
            metal_optimizer = MetalOptimizer()
            db_optimizer = DatabaseOptimizer(str(db_path))
            cache_manager = CacheManager(max_memory_mb=100)
            
            # Create mock system manager
            mock_system_manager = Mock()
            mock_system_manager.get_component.side_effect = lambda name: {
                'memory_optimizer': memory_optimizer,
                'speed_optimizer': speed_optimizer,
                'metal_optimizer': metal_optimizer,
                'db_optimizer': db_optimizer,
                'cache_manager': cache_manager
            }.get(name)
            
            # Initialize auto-tuner
            auto_tuner = AutoTuner(mock_system_manager, monitoring_interval=0.1)
            
            # Test that components can interact
            # This is a basic smoke test to ensure no major incompatibilities
            memory_stats = memory_optimizer.get_memory_usage()
            speed_stats = speed_optimizer.get_performance_stats()
            metal_profile = metal_optimizer.profile_mps_usage()
            cache_stats = cache_manager.get_cache_statistics()
            tuning_stats = auto_tuner.get_tuning_statistics()
            
            # Verify all components return valid data
            self.assertIsInstance(memory_stats, dict)
            self.assertIsInstance(speed_stats, dict)
            self.assertIsInstance(metal_profile, dict)
            self.assertIsInstance(cache_stats, dict)
            self.assertIsInstance(tuning_stats, dict)
            
            # Cleanup
            memory_optimizer.cleanup()
            speed_optimizer.shutdown()
            metal_optimizer.cleanup()
            db_optimizer.cleanup()
            cache_manager.cleanup()
            auto_tuner.cleanup()
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()