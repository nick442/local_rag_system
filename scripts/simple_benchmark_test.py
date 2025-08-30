#!/usr/bin/env python3
"""
Simple benchmark test to identify issues before running the complex runner.

This script will:
1. Test basic RAG pipeline functionality
2. Identify import issues
3. Test minimal benchmark operations
4. Report what works and what doesn't

This helps us fix the complex benchmark runner by identifying specific failure points.
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

def test_imports():
    """Test that all required imports work"""
    print("🔍 Testing imports...")
    
    try:
        from src.rag_pipeline import RAGPipeline
        print("  ✓ RAGPipeline import successful")
        return True
    except Exception as e:
        print(f"  ❌ RAGPipeline import failed: {e}")
        return False

def test_rag_pipeline_basic():
    """Test basic RAG pipeline functionality"""
    print("🤖 Testing RAG pipeline basic functionality...")
    
    try:
        from src.rag_pipeline import RAGPipeline
        
        # Default paths from main.py
        DEFAULT_DB_PATH = "data/rag_vectors.db"
        DEFAULT_EMBEDDING_PATH = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        DEFAULT_LLM_PATH = "models/gemma-3-4b-it-q4_0.gguf"
        
        # Try to initialize pipeline
        rag = RAGPipeline(
            db_path=DEFAULT_DB_PATH,
            embedding_model_path=DEFAULT_EMBEDDING_PATH,
            llm_model_path=DEFAULT_LLM_PATH
        )
        print("  ✓ RAG pipeline initialization successful")
        
        # Try a simple query
        print("  🔄 Testing simple query...")
        start_time = time.time()
        response = rag.query("What is machine learning?", k=3)
        latency = time.time() - start_time
        
        print(f"  ✓ Query successful (latency: {latency:.2f}s)")
        
        # Check response structure
        if isinstance(response, dict):
            print(f"    - Response type: dict")
            print(f"    - Keys: {list(response.keys())}")
            if 'response' in response:
                response_text = response['response']
                print(f"    - Response length: {len(response_text)} chars")
            if 'sources' in response:
                sources = response['sources']
                print(f"    - Sources count: {len(sources)}")
        else:
            print(f"    - Response type: {type(response)}")
            print(f"    - Response length: {len(str(response))} chars")
        
        return True, rag, response, latency
        
    except Exception as e:
        print(f"  ❌ RAG pipeline test failed: {e}")
        return False, None, None, 0

def test_benchmark_imports():
    """Test benchmark module imports"""
    print("📊 Testing benchmark imports...")
    
    results = {}
    
    try:
        from benchmarks.performance_suite import PerformanceBenchmark
        print("  ✓ PerformanceBenchmark import successful")
        results['performance'] = True
    except Exception as e:
        print(f"  ❌ PerformanceBenchmark import failed: {e}")
        results['performance'] = False
    
    try:
        from benchmarks.accuracy_suite import AccuracyBenchmark
        print("  ✓ AccuracyBenchmark import successful")
        results['accuracy'] = True
    except Exception as e:
        print(f"  ❌ AccuracyBenchmark import failed: {e}")
        results['accuracy'] = False
    
    try:
        from benchmarks.monitoring import RAGMonitor
        print("  ✓ RAGMonitor import successful")
        results['monitoring'] = True
    except Exception as e:
        print(f"  ❌ RAGMonitor import failed: {e}")
        results['monitoring'] = False
    
    return results

def test_simple_performance_benchmark(rag):
    """Test simple performance benchmark"""
    print("⚡ Testing simple performance benchmark...")
    
    try:
        from benchmarks.performance_suite import PerformanceBenchmark
        
        # Initialize benchmark
        benchmark = PerformanceBenchmark()
        print("  ✓ Performance benchmark initialized")
        
        # Test memory measurement
        memory_info = benchmark._get_memory_usage()
        print(f"  ✓ Memory measurement: {memory_info['rss_mb']:.2f} MB RSS")
        
        # Test token counting
        token_count = benchmark._count_tokens("This is a test sentence")
        print(f"  ✓ Token counting: {token_count} tokens")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance benchmark test failed: {e}")
        return False

def test_simple_accuracy_benchmark(rag):
    """Test simple accuracy benchmark"""
    print("📈 Testing simple accuracy benchmark...")
    
    try:
        from benchmarks.accuracy_suite import AccuracyBenchmark
        
        # Initialize benchmark  
        benchmark = AccuracyBenchmark()
        print("  ✓ Accuracy benchmark initialized")
        
        # Test text similarity
        sim = benchmark._calculate_text_similarity("machine learning", "learning machines")
        print(f"  ✓ Text similarity calculation: {sim:.3f}")
        
        # Test keyword extraction
        keywords = benchmark._extract_keywords("machine learning algorithms process data")
        print(f"  ✓ Keyword extraction: {keywords}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Accuracy benchmark test failed: {e}")
        return False

def main():
    """Run all simple tests"""
    print("🧪 Simple Benchmark Test Suite")
    print("=" * 50)
    
    results = {
        'imports_working': False,
        'rag_pipeline_working': False,
        'benchmark_imports_working': {},
        'performance_benchmark_working': False,
        'accuracy_benchmark_working': False
    }
    
    # Test basic imports
    results['imports_working'] = test_imports()
    
    if not results['imports_working']:
        print("\n💥 Basic imports failed - need to fix RAG pipeline setup first")
        return results
    
    # Test RAG pipeline
    rag_success, rag, response, latency = test_rag_pipeline_basic()
    results['rag_pipeline_working'] = rag_success
    
    if not rag_success:
        print("\n💥 RAG pipeline not working - need to fix pipeline setup")
        return results
    
    # Test benchmark imports
    results['benchmark_imports_working'] = test_benchmark_imports()
    
    # Test performance benchmark if import worked
    if results['benchmark_imports_working'].get('performance', False):
        results['performance_benchmark_working'] = test_simple_performance_benchmark(rag)
    
    # Test accuracy benchmark if import worked
    if results['benchmark_imports_working'].get('accuracy', False):
        results['accuracy_benchmark_working'] = test_simple_accuracy_benchmark(rag)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"  • Basic imports: {'✅' if results['imports_working'] else '❌'}")
    print(f"  • RAG pipeline: {'✅' if results['rag_pipeline_working'] else '❌'}")
    print(f"  • Performance benchmark: {'✅' if results['performance_benchmark_working'] else '❌'}")
    print(f"  • Accuracy benchmark: {'✅' if results['accuracy_benchmark_working'] else '❌'}")
    
    all_working = (
        results['imports_working'] and 
        results['rag_pipeline_working'] and
        results['performance_benchmark_working'] and
        results['accuracy_benchmark_working']
    )
    
    if all_working:
        print("\n🎉 All basic tests passed! Complex benchmark runner should work.")
        print("💡 Next step: Run the full benchmark suite with:")
        print("   python scripts/run_benchmarks.py --performance --corpus-size small")
    else:
        print("\n⚠️  Some tests failed. Issues to fix:")
        if not results['imports_working']:
            print("   - Fix RAG pipeline import/setup issues")
        if not results['rag_pipeline_working']:
            print("   - Fix RAG pipeline functionality")
        if not results['performance_benchmark_working']:
            print("   - Fix performance benchmark initialization")
        if not results['accuracy_benchmark_working']:
            print("   - Fix accuracy benchmark initialization")
        print("\n💡 Fix these issues before running the complex benchmark runner.")
    
    return results

if __name__ == "__main__":
    main()