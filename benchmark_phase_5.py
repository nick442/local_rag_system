"""
Performance Benchmarks for Phase 5: LLM Integration
Measures performance metrics for the complete RAG pipeline.
"""

import os
import sys
import time
import yaml
import psutil
from pathlib import Path
from typing import Dict, Any, List
import statistics

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm_wrapper import LLMWrapper, create_llm_wrapper
from src.prompt_builder import PromptBuilder, create_prompt_builder
from src.rag_pipeline import RAGPipeline, create_rag_pipeline
from src.retriever import RetrievalResult


class RAGBenchmark:
    """Comprehensive benchmarking suite for RAG pipeline."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize benchmark with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        
        # Test queries for benchmarking
        self.test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain deep learning algorithms",
            "What are the applications of artificial intelligence?",
            "How does natural language processing work?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain backpropagation in neural networks",
            "What are transformer models?",
            "How does gradient descent optimization work?",
            "What is reinforcement learning?"
        ]
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            print(f"⚠️  Config file not found: {self.config_path}")
            return {}
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def benchmark_llm_loading(self) -> Dict[str, Any]:
        """Benchmark model loading time and memory usage."""
        print("\n🔄 Benchmarking LLM loading...")
        
        # Get model path
        model_path = self.config.get('model', {}).get('path')
        if not model_path or not os.path.exists(model_path):
            print(f"❌ Model not found at: {model_path}")
            return {'error': 'Model file not found'}
        
        # Measure loading time and memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        try:
            llm_params = self.config.get('llm_params', {})
            wrapper = LLMWrapper(model_path, **llm_params)
            load_time = time.time() - start_time
            
            # Measure memory after loading
            loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = loaded_memory - initial_memory
            
            # Get model info
            model_info = wrapper.get_model_info()
            
            wrapper.unload_model()
            
            results = {
                'load_time_seconds': load_time,
                'memory_usage_mb': memory_increase,
                'model_info': model_info,
                'success': True
            }
            
            print(f"✅ Model loaded in {load_time:.2f}s, using {memory_increase:.1f}MB")
            return results
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def benchmark_generation_speed(self, num_tests: int = 5) -> Dict[str, Any]:
        """Benchmark text generation speed."""
        print(f"\n🔄 Benchmarking generation speed ({num_tests} tests)...")
        
        model_path = self.config.get('model', {}).get('path')
        if not model_path or not os.path.exists(model_path):
            return {'error': 'Model file not found'}
        
        try:
            llm_params = self.config.get('llm_params', {})
            wrapper = LLMWrapper(model_path, **llm_params)
            
            generation_times = []
            first_token_latencies = []
            tokens_per_second_list = []
            total_tokens = 0
            
            for i in range(num_tests):
                query = self.test_queries[i % len(self.test_queries)]
                prompt = f"<bos><start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
                
                print(f"  Test {i+1}/{num_tests}: '{query[:30]}...'")
                
                # Measure generation with streaming for first token latency
                start_time = time.time()
                first_token_time = None
                generated_tokens = 0
                generated_text = ""
                
                for token in wrapper.generate_stream(prompt, max_tokens=100, temperature=0.7):
                    if first_token_time is None:
                        first_token_time = time.time()
                        first_token_latencies.append(first_token_time - start_time)
                    
                    generated_tokens += 1
                    generated_text += token
                
                end_time = time.time()
                total_time = end_time - start_time
                
                if generated_tokens > 0:
                    tokens_per_second = generated_tokens / total_time
                    tokens_per_second_list.append(tokens_per_second)
                
                generation_times.append(total_time)
                total_tokens += generated_tokens
                
                print(f"    Generated {generated_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
            
            wrapper.unload_model()
            
            # Calculate statistics
            results = {
                'num_tests': num_tests,
                'total_tokens_generated': total_tokens,
                'avg_generation_time': statistics.mean(generation_times),
                'avg_first_token_latency_ms': statistics.mean(first_token_latencies) * 1000,
                'avg_tokens_per_second': statistics.mean(tokens_per_second_list),
                'min_tokens_per_second': min(tokens_per_second_list),
                'max_tokens_per_second': max(tokens_per_second_list),
                'generation_times': generation_times,
                'tokens_per_second_values': tokens_per_second_list,
                'success': True
            }
            
            print(f"✅ Average: {results['avg_tokens_per_second']:.1f} tok/s, "
                  f"First token: {results['avg_first_token_latency_ms']:.1f}ms")
            
            return results
            
        except Exception as e:
            print(f"❌ Generation benchmark failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def benchmark_rag_pipeline(self, num_tests: int = 3) -> Dict[str, Any]:
        """Benchmark complete RAG pipeline end-to-end."""
        print(f"\n🔄 Benchmarking RAG pipeline ({num_tests} tests)...")
        
        # Check required paths
        paths_to_check = {
            'db_path': 'data/rag_vectors.db',
            'embedding_model_path': 'models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf',
            'llm_model_path': self.config.get('model', {}).get('path')
        }
        
        missing_paths = []
        for name, path in paths_to_check.items():
            if not path or not os.path.exists(path):
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            print(f"❌ Required files missing: {', '.join(missing_paths)}")
            return {
                'error': f"Missing files: {', '.join(missing_paths)}",
                'success': False
            }
        
        try:
            # Initialize pipeline
            pipeline_start = time.time()
            pipeline = RAGPipeline(
                db_path=paths_to_check['db_path'],
                embedding_model_path=paths_to_check['embedding_model_path'],
                llm_model_path=paths_to_check['llm_model_path'],
                config_path=self.config_path
            )
            pipeline_init_time = time.time() - pipeline_start
            
            end_to_end_times = []
            retrieval_times = []
            generation_times = []
            total_tokens_generated = 0
            
            for i in range(num_tests):
                query = self.test_queries[i]
                print(f"  Test {i+1}/{num_tests}: '{query[:40]}...'")
                
                start_time = time.time()
                
                try:
                    response = pipeline.query(
                        query, 
                        k=3, 
                        stream=False,
                        max_tokens=100,
                        temperature=0.7
                    )
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Extract timing information
                    metadata = response.get('metadata', {})
                    retrieval_time = metadata.get('retrieval_time', 0)
                    generation_time = metadata.get('generation_time', 0)
                    output_tokens = metadata.get('output_tokens', 0)
                    tokens_per_second = metadata.get('tokens_per_second', 0)
                    
                    end_to_end_times.append(total_time)
                    retrieval_times.append(retrieval_time)
                    generation_times.append(generation_time)
                    total_tokens_generated += output_tokens
                    
                    print(f"    Total: {total_time:.2f}s, "
                          f"Retrieval: {retrieval_time:.3f}s, "
                          f"Generation: {generation_time:.2f}s "
                          f"({tokens_per_second:.1f} tok/s)")
                
                except Exception as e:
                    print(f"    ❌ Query failed: {e}")
                    continue
            
            # Calculate statistics
            if end_to_end_times:
                results = {
                    'pipeline_init_time': pipeline_init_time,
                    'num_successful_tests': len(end_to_end_times),
                    'total_tokens_generated': total_tokens_generated,
                    'avg_end_to_end_time': statistics.mean(end_to_end_times),
                    'avg_retrieval_time': statistics.mean(retrieval_times),
                    'avg_generation_time': statistics.mean(generation_times),
                    'min_end_to_end_time': min(end_to_end_times),
                    'max_end_to_end_time': max(end_to_end_times),
                    'pipeline_stats': pipeline.get_stats(),
                    'success': True
                }
                
                print(f"✅ Average E2E: {results['avg_end_to_end_time']:.2f}s, "
                      f"Retrieval: {results['avg_retrieval_time']:.3f}s, "
                      f"Generation: {results['avg_generation_time']:.2f}s")
            else:
                results = {
                    'error': 'No successful tests',
                    'success': False
                }
            
            return results
            
        except Exception as e:
            print(f"❌ RAG pipeline benchmark failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def benchmark_streaming_performance(self, num_tests: int = 3) -> Dict[str, Any]:
        """Benchmark streaming generation performance."""
        print(f"\n🔄 Benchmarking streaming performance ({num_tests} tests)...")
        
        model_path = self.config.get('model', {}).get('path')
        if not model_path or not os.path.exists(model_path):
            return {'error': 'Model file not found'}
        
        try:
            llm_params = self.config.get('llm_params', {})
            wrapper = LLMWrapper(model_path, **llm_params)
            
            streaming_results = []
            
            for i in range(num_tests):
                query = self.test_queries[i]
                prompt = f"<bos><start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
                
                print(f"  Test {i+1}/{num_tests}: '{query[:30]}...'")
                
                start_time = time.time()
                first_token_time = None
                token_times = []
                generated_tokens = 0
                
                for token in wrapper.generate_stream(prompt, max_tokens=50, temperature=0.7):
                    current_time = time.time()
                    
                    if first_token_time is None:
                        first_token_time = current_time
                    else:
                        token_times.append(current_time - token_times[-1] if token_times else current_time - first_token_time)
                    
                    generated_tokens += 1
                
                end_time = time.time()
                total_time = end_time - start_time
                first_token_latency = first_token_time - start_time if first_token_time else 0
                
                if generated_tokens > 0:
                    avg_inter_token_latency = statistics.mean(token_times) if token_times else 0
                    tokens_per_second = generated_tokens / total_time
                    
                    result = {
                        'query': query,
                        'total_time': total_time,
                        'first_token_latency_ms': first_token_latency * 1000,
                        'avg_inter_token_latency_ms': avg_inter_token_latency * 1000,
                        'tokens_generated': generated_tokens,
                        'tokens_per_second': tokens_per_second
                    }
                    
                    streaming_results.append(result)
                    
                    print(f"    {generated_tokens} tokens, "
                          f"First: {first_token_latency*1000:.1f}ms, "
                          f"Rate: {tokens_per_second:.1f} tok/s")
            
            wrapper.unload_model()
            
            if streaming_results:
                # Calculate aggregate statistics
                results = {
                    'num_tests': len(streaming_results),
                    'avg_first_token_latency_ms': statistics.mean([r['first_token_latency_ms'] for r in streaming_results]),
                    'avg_inter_token_latency_ms': statistics.mean([r['avg_inter_token_latency_ms'] for r in streaming_results]),
                    'avg_tokens_per_second': statistics.mean([r['tokens_per_second'] for r in streaming_results]),
                    'total_tokens_generated': sum([r['tokens_generated'] for r in streaming_results]),
                    'individual_results': streaming_results,
                    'success': True
                }
                
                print(f"✅ Streaming avg: First token {results['avg_first_token_latency_ms']:.1f}ms, "
                      f"Inter-token {results['avg_inter_token_latency_ms']:.1f}ms")
                
                return results
            else:
                return {'error': 'No successful streaming tests', 'success': False}
                
        except Exception as e:
            print(f"❌ Streaming benchmark failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("\n🔄 Benchmarking memory usage...")
        
        model_path = self.config.get('model', {}).get('path')
        if not model_path or not os.path.exists(model_path):
            return {'error': 'Model file not found'}
        
        try:
            process = psutil.Process()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load model
            llm_params = self.config.get('llm_params', {})
            wrapper = LLMWrapper(model_path, **llm_params)
            loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate some text to measure generation memory
            test_query = "What is artificial intelligence and how does it work?"
            prompt = f"<bos><start_of_turn>user\n{test_query}<end_of_turn>\n<start_of_turn>model\n"
            
            pre_generation_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            _ = wrapper.generate(prompt, max_tokens=200, temperature=0.7)
            
            post_generation_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Unload model
            wrapper.unload_model()
            unloaded_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            results = {
                'baseline_memory_mb': baseline_memory,
                'model_loading_increase_mb': loaded_memory - baseline_memory,
                'generation_increase_mb': post_generation_memory - pre_generation_memory,
                'memory_after_unload_mb': unloaded_memory,
                'peak_memory_mb': post_generation_memory,
                'memory_recovered_mb': post_generation_memory - unloaded_memory,
                'success': True
            }
            
            print(f"✅ Memory: Baseline {baseline_memory:.1f}MB, "
                  f"Peak {post_generation_memory:.1f}MB, "
                  f"Model loading +{results['model_loading_increase_mb']:.1f}MB")
            
            return results
            
        except Exception as e:
            print(f"❌ Memory benchmark failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        print("🚀 Starting comprehensive RAG pipeline benchmark")
        print("="*60)
        
        benchmark_start = time.time()
        
        # Run all benchmarks
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_path': self.config_path,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'platform': sys.platform
            }
        }
        
        # Model loading benchmark
        self.results['model_loading'] = self.benchmark_llm_loading()
        
        # Generation speed benchmark
        self.results['generation_speed'] = self.benchmark_generation_speed()
        
        # Streaming performance benchmark
        self.results['streaming_performance'] = self.benchmark_streaming_performance()
        
        # Memory usage benchmark
        self.results['memory_usage'] = self.benchmark_memory_usage()
        
        # RAG pipeline benchmark (if components available)
        self.results['rag_pipeline'] = self.benchmark_rag_pipeline()
        
        total_time = time.time() - benchmark_start
        self.results['benchmark_duration'] = total_time
        
        print("\n" + "="*60)
        print(f"🎉 Benchmark completed in {total_time:.2f}s")
        
        return self.results
    
    def save_results(self, output_path: str = "benchmark_results.yaml"):
        """Save benchmark results to file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False, indent=2)
        print(f"📊 Results saved to: {output_path}")
    
    def print_summary(self):
        """Print benchmark summary."""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY")
        print("="*60)
        
        # Model loading
        if 'model_loading' in self.results and self.results['model_loading'].get('success'):
            ml = self.results['model_loading']
            print(f"🔄 Model Loading: {ml['load_time_seconds']:.2f}s, {ml['memory_usage_mb']:.1f}MB")
        
        # Generation speed
        if 'generation_speed' in self.results and self.results['generation_speed'].get('success'):
            gs = self.results['generation_speed']
            print(f"⚡ Generation Speed: {gs['avg_tokens_per_second']:.1f} tok/s avg, "
                  f"{gs['avg_first_token_latency_ms']:.1f}ms first token")
        
        # Streaming
        if 'streaming_performance' in self.results and self.results['streaming_performance'].get('success'):
            sp = self.results['streaming_performance']
            print(f"📡 Streaming: {sp['avg_tokens_per_second']:.1f} tok/s, "
                  f"{sp['avg_first_token_latency_ms']:.1f}ms first token")
        
        # Memory
        if 'memory_usage' in self.results and self.results['memory_usage'].get('success'):
            mu = self.results['memory_usage']
            print(f"🧠 Memory: Peak {mu['peak_memory_mb']:.1f}MB, "
                  f"Model +{mu['model_loading_increase_mb']:.1f}MB")
        
        # RAG Pipeline
        if 'rag_pipeline' in self.results and self.results['rag_pipeline'].get('success'):
            rp = self.results['rag_pipeline']
            print(f"🔗 RAG Pipeline: {rp['avg_end_to_end_time']:.2f}s avg E2E, "
                  f"{rp['avg_retrieval_time']:.3f}s retrieval")
        
        print("="*60)


def main():
    """Main benchmark execution."""
    benchmark = RAGBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    benchmark.save_results("benchmark_phase_5_results.yaml")
    
    # Print summary
    benchmark.print_summary()
    
    # Return success status
    successful_tests = sum(1 for test_name, test_result in results.items() 
                          if isinstance(test_result, dict) and test_result.get('success', False))
    
    print(f"\n✅ {successful_tests} benchmark tests completed successfully")
    
    return results


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)