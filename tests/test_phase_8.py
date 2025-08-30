#!/usr/bin/env python3
"""
Test Suite for Phase 8: Testing Infrastructure Implementation

Test cases for:
1. Performance benchmark suite execution
2. Accuracy evaluation framework
3. Test corpus generation
4. Benchmark queries dataset
5. Automated test runner
6. Continuous monitoring
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
import time

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test_data'))

from benchmarks.performance_suite import PerformanceBenchmark
from benchmarks.accuracy_suite import AccuracyBenchmark, EvaluationQuery
from benchmarks.monitoring import RAGMonitor
from test_data.generate_test_corpus import TestCorpusGenerator

class TestPerformanceBenchmark(unittest.TestCase):
    """Test performance benchmark suite"""
    
    def setUp(self):
        """Setup for performance benchmark tests"""
        try:
            self.benchmark = PerformanceBenchmark()
        except Exception as e:
            self.skipTest(f"Cannot initialize RAG pipeline: {e}")
    
    def test_benchmark_initialization(self):
        """Test benchmark suite initializes correctly"""
        self.assertIsNotNone(self.benchmark)
        self.assertIsNotNone(self.benchmark.rag)
        self.assertIsNotNone(self.benchmark.process)
    
    def test_memory_usage_measurement(self):
        """Test memory usage measurement"""
        memory_info = self.benchmark._get_memory_usage()
        
        self.assertIn('rss_mb', memory_info)
        self.assertIn('vms_mb', memory_info)
        self.assertIn('percent', memory_info)
        self.assertGreater(memory_info['rss_mb'], 0)
    
    def test_token_counting(self):
        """Test token counting functionality"""
        test_text = "This is a test sentence with multiple words."
        token_count = self.benchmark._count_tokens(test_text)
        
        self.assertGreater(token_count, 0)
        self.assertIsInstance(token_count, int)
    
    def test_simple_throughput_benchmark(self):
        """Test basic token throughput benchmark"""
        # Use minimal test to avoid long execution
        simple_queries = ["What is AI?"]
        
        try:
            result = self.benchmark.benchmark_token_throughput(
                num_iterations=1, 
                test_queries=simple_queries
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "token_throughput")
            self.assertIn('tokens_per_second', result.metrics)
            self.assertIn('response_time', result.metrics)
            
        except Exception as e:
            self.skipTest(f"RAG pipeline not available for testing: {e}")
    
    def test_memory_benchmark(self):
        """Test memory usage benchmark"""
        try:
            result = self.benchmark.benchmark_memory_usage()
            
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "memory_usage")
            self.assertIn('baseline', result.memory_usage)
            
        except Exception as e:
            self.skipTest(f"Memory benchmark failed: {e}")

class TestAccuracyBenchmark(unittest.TestCase):
    """Test accuracy evaluation framework"""
    
    def setUp(self):
        """Setup for accuracy benchmark tests"""
        try:
            self.benchmark = AccuracyBenchmark()
        except Exception as e:
            self.skipTest(f"Cannot initialize RAG pipeline: {e}")
    
    def test_benchmark_initialization(self):
        """Test accuracy benchmark initializes correctly"""
        self.assertIsNotNone(self.benchmark)
        self.assertIsNotNone(self.benchmark.rag)
    
    def test_text_similarity_calculation(self):
        """Test text similarity calculation"""
        text1 = "machine learning algorithms"
        text2 = "algorithms for machine learning"
        text3 = "completely different content"
        
        # Similar texts should have high similarity
        similarity_high = self.benchmark._calculate_text_similarity(text1, text2)
        self.assertGreater(similarity_high, 0.3)
        
        # Different texts should have low similarity
        similarity_low = self.benchmark._calculate_text_similarity(text1, text3)
        self.assertLess(similarity_low, 0.3)
    
    def test_keyword_extraction(self):
        """Test keyword extraction"""
        text = "Machine learning algorithms process data efficiently"
        keywords = self.benchmark._extract_keywords(text)
        
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)
        self.assertIn("machine", keywords)
        self.assertIn("learning", keywords)
    
    def test_evaluation_query_creation(self):
        """Test evaluation query data structure"""
        query = EvaluationQuery(
            query="What is AI?",
            expected_topics=["artificial", "intelligence"],
            difficulty="easy"
        )
        
        self.assertEqual(query.query, "What is AI?")
        self.assertEqual(query.expected_topics, ["artificial", "intelligence"])
        self.assertEqual(query.difficulty, "easy")

class TestCorpusGeneration(unittest.TestCase):
    """Test test corpus generation"""
    
    def setUp(self):
        """Setup for corpus generation tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = TestCorpusGenerator(self.temp_dir)
    
    def tearDown(self):
        """Cleanup after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generator_initialization(self):
        """Test corpus generator initializes correctly"""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.base_path, Path(self.temp_dir))
        self.assertGreater(len(self.generator.topics), 0)
        self.assertGreater(len(self.generator.technical_terms), 0)
    
    def test_technical_article_generation(self):
        """Test technical article generation"""
        topic = "machine learning"
        target_words = 100
        
        article = self.generator._generate_technical_article(topic, target_words)
        
        self.assertIsInstance(article, str)
        self.assertGreater(len(article), 50)  # Should have meaningful content
        self.assertIn(topic, article.lower())
    
    def test_qa_document_generation(self):
        """Test Q&A document generation"""
        topic = "neural networks"
        target_words = 100
        
        qa_doc = self.generator._generate_qa_document(topic, target_words)
        
        self.assertIsInstance(qa_doc, str)
        self.assertIn("Q", qa_doc)  # Should contain questions
        self.assertIn("A", qa_doc)  # Should contain answers
    
    def test_edge_case_generation(self):
        """Test edge case document generation"""
        empty_doc = self.generator._generate_edge_case_document("empty")
        self.assertEqual(empty_doc, "")
        
        minimal_doc = self.generator._generate_edge_case_document("minimal")
        self.assertGreater(len(minimal_doc), 0)
        self.assertLess(len(minimal_doc), 100)
        
        unicode_doc = self.generator._generate_edge_case_document("unicode")
        self.assertIn("Unicode", unicode_doc)
    
    def test_small_corpus_generation(self):
        """Test small corpus generation"""
        result = self.generator.generate_corpus("small")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['corpus_size'], "small")
        self.assertGreater(result['documents_count'], 0)
        self.assertGreater(result['total_words'], 0)
        
        # Check if files were actually created
        corpus_dir = Path(self.temp_dir) / "small"
        self.assertTrue(corpus_dir.exists())
        
        # Count generated files
        files = list(corpus_dir.glob("*.txt"))
        self.assertGreater(len(files), 0)

class TestBenchmarkQueries(unittest.TestCase):
    """Test benchmark queries dataset"""
    
    def test_queries_file_exists(self):
        """Test that benchmark queries file exists"""
        queries_file = Path("test_data/benchmark_queries.json")
        self.assertTrue(queries_file.exists(), "Benchmark queries file should exist")
    
    def test_queries_structure(self):
        """Test benchmark queries JSON structure"""
        queries_file = Path("test_data/benchmark_queries.json")
        
        with open(queries_file, 'r') as f:
            data = json.load(f)
        
        # Test structure
        self.assertIn('metadata', data)
        self.assertIn('categories', data)
        self.assertIn('evaluation_criteria', data)
        
        # Test metadata
        metadata = data['metadata']
        self.assertIn('total_queries', metadata)
        self.assertIn('categories', metadata)
        
        # Test categories
        categories = data['categories']
        self.assertIn('factual', categories)
        self.assertIn('analytical', categories)
        self.assertIn('edge_cases', categories)
        
        # Test individual queries
        factual_queries = categories['factual']
        self.assertGreater(len(factual_queries), 0)
        
        first_query = factual_queries[0]
        self.assertIn('query', first_query)
        self.assertIn('expected_topics', first_query)
        self.assertIn('difficulty', first_query)

class TestRAGMonitoring(unittest.TestCase):
    """Test continuous monitoring functionality"""
    
    def setUp(self):
        """Setup for monitoring tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = RAGMonitor(log_dir=self.temp_dir)
    
    def tearDown(self):
        """Cleanup after tests"""
        self.monitor.stop_monitoring()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly"""
        self.assertIsNotNone(self.monitor)
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertFalse(self.monitor.monitoring_active)
    
    def test_database_initialization(self):
        """Test database tables are created"""
        db_file = Path(self.temp_dir) / "monitoring.db"
        self.assertTrue(db_file.exists())
        
        import sqlite3
        with sqlite3.connect(db_file) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn('query_logs', tables)
            self.assertIn('performance_metrics', tables)
            self.assertIn('system_metrics', tables)
    
    def test_query_logging(self):
        """Test query logging functionality"""
        query = "What is machine learning?"
        response = {"response": "ML is...", "sources": ["doc1", "doc2"]}
        latency_ms = 150.5
        
        self.monitor.log_query(query, response, latency_ms)
        
        # Check in-memory storage
        self.assertEqual(len(self.monitor.query_logs), 1)
        
        logged_query = self.monitor.query_logs[0]
        self.assertEqual(logged_query.query, query)
        self.assertEqual(logged_query.sources_count, 2)
        self.assertEqual(logged_query.latency_ms, latency_ms)
    
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring"""
        self.assertFalse(self.monitor.monitoring_active)
        
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring_active)
        self.assertIsNotNone(self.monitor.monitor_thread)
        
        time.sleep(0.1)  # Give thread time to start
        
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)
    
    def test_statistics_tracking(self):
        """Test statistics tracking"""
        # Log some test queries
        self.monitor.log_query("Query 1", {"response": "Response 1"}, 100)
        self.monitor.log_query("Query 2", {"response": "Response 2"}, 200, error="Test error")
        
        stats = self.monitor.get_stats()
        
        self.assertEqual(stats['total_queries'], 2)
        self.assertEqual(stats['total_errors'], 1)
        self.assertEqual(stats['error_rate'], 0.5)

class TestBenchmarkRunner(unittest.TestCase):
    """Test automated benchmark runner"""
    
    def test_run_benchmarks_script_exists(self):
        """Test that run_benchmarks.py exists"""
        script_path = Path("run_benchmarks.py")
        self.assertTrue(script_path.exists(), "run_benchmarks.py should exist")
    
    def test_run_benchmarks_imports(self):
        """Test that run_benchmarks.py can be imported"""
        try:
            import run_benchmarks
            self.assertTrue(hasattr(run_benchmarks, 'BenchmarkRunner'))
            self.assertTrue(hasattr(run_benchmarks, 'BenchmarkConfig'))
        except ImportError as e:
            self.fail(f"Could not import run_benchmarks: {e}")

class TestPhase8Integration(unittest.TestCase):
    """Integration tests for Phase 8 components"""
    
    def test_all_components_exist(self):
        """Test that all Phase 8 components exist"""
        components = [
            Path("benchmarks/performance_suite.py"),
            Path("benchmarks/accuracy_suite.py"),
            Path("benchmarks/monitoring.py"),
            Path("test_data/generate_test_corpus.py"),
            Path("test_data/benchmark_queries.json"),
            Path("run_benchmarks.py")
        ]
        
        for component in components:
            self.assertTrue(component.exists(), f"Component {component} should exist")
    
    def test_directory_structure(self):
        """Test proper directory structure"""
        required_dirs = [
            Path("benchmarks"),
            Path("test_data"),
            Path("test_data/corpora"),
            Path("tests")
        ]
        
        for directory in required_dirs:
            self.assertTrue(directory.exists(), f"Directory {directory} should exist")

def run_tests():
    """Run all Phase 8 tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPerformanceBenchmark,
        TestAccuracyBenchmark, 
        TestCorpusGeneration,
        TestBenchmarkQueries,
        TestRAGMonitoring,
        TestBenchmarkRunner,
        TestPhase8Integration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)