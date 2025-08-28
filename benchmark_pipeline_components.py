#!/usr/bin/env python3
"""
Comprehensive pipeline component benchmarking tool.
Measures performance of every component in the JSON to vector database pipeline.
"""

import time
import logging
import sys
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class PipelineBenchmarker:
    """Comprehensive benchmarking tool for all pipeline components."""
    
    def __init__(self, test_dir: str = "data/test_100"):
        self.test_dir = Path(test_dir)
        self.results = {}
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024
        }
    
    def benchmark_file_loading(self, sample_files: List[Path]) -> Dict[str, Any]:
        """Benchmark raw file loading performance."""
        print("\n=== BENCHMARKING FILE LOADING ===")
        
        start_memory = self.get_memory_usage()
        load_times = []
        file_sizes = []
        
        for file_path in sample_files:
            start_time = time.time()
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                load_time = time.time() - start_time
                load_times.append(load_time)
                file_sizes.append(len(content))
                
            except Exception as e:
                print(f"ERROR loading {file_path}: {e}")
                continue
        
        end_memory = self.get_memory_usage()
        
        results = {
            'files_tested': len(load_times),
            'avg_load_time': statistics.mean(load_times) if load_times else 0,
            'total_load_time': sum(load_times),
            'avg_file_size': statistics.mean(file_sizes) if file_sizes else 0,
            'total_chars_loaded': sum(file_sizes),
            'chars_per_second': sum(file_sizes) / sum(load_times) if load_times else 0,
            'memory_delta_mb': end_memory['rss_mb'] - start_memory['rss_mb']
        }
        
        print(f"File Loading Results:")
        print(f"  Files: {results['files_tested']}")
        print(f"  Avg load time: {results['avg_load_time']:.4f}s")
        print(f"  Speed: {results['chars_per_second']:.0f} chars/sec")
        print(f"  Memory delta: {results['memory_delta_mb']:.1f} MB")
        
        return results
    
    def benchmark_document_ingestion(self, sample_files: List[Path]) -> Dict[str, Any]:
        """Benchmark DocumentIngestionService component by component."""
        print("\n=== BENCHMARKING DOCUMENT INGESTION ===")
        
        from src.document_ingestion import DocumentIngestionService, TextLoader, DocumentChunker
        
        # Test TextLoader
        print("\n--- TextLoader Performance ---")
        loader = TextLoader()
        start_memory = self.get_memory_usage()
        load_times = []
        
        documents = []
        for file_path in sample_files:
            start_time = time.time()
            try:
                document = loader.load(str(file_path))
                load_time = time.time() - start_time
                load_times.append(load_time)
                documents.append(document)
            except Exception as e:
                print(f"ERROR loading {file_path}: {e}")
                continue
        
        loader_memory = self.get_memory_usage()
        
        # Test DocumentChunker
        print("\n--- DocumentChunker Performance ---")
        chunker = DocumentChunker(chunk_size=512, overlap=128)
        chunk_times = []
        chunk_counts = []
        all_chunks = []
        
        for doc in documents:
            start_time = time.time()
            try:
                chunks = chunker.chunk_document(doc)
                chunk_time = time.time() - start_time
                chunk_times.append(chunk_time)
                chunk_counts.append(len(chunks))
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"ERROR chunking document: {e}")
                continue
        
        chunker_memory = self.get_memory_usage()
        
        # Full DocumentIngestionService
        print("\n--- Full DocumentIngestionService ---")
        service = DocumentIngestionService(chunk_size=512, chunk_overlap=128)
        service_times = []
        service_chunks = []
        
        for file_path in sample_files:
            start_time = time.time()
            try:
                chunks = service.ingest_document(str(file_path))
                service_time = time.time() - start_time
                service_times.append(service_time)
                service_chunks.append(len(chunks))
            except Exception as e:
                print(f"ERROR with service ingestion {file_path}: {e}")
                continue
        
        end_memory = self.get_memory_usage()
        
        results = {
            'text_loader': {
                'avg_time': statistics.mean(load_times) if load_times else 0,
                'total_time': sum(load_times),
                'docs_loaded': len(documents),
                'memory_delta_mb': loader_memory['rss_mb'] - start_memory['rss_mb']
            },
            'document_chunker': {
                'avg_time': statistics.mean(chunk_times) if chunk_times else 0,
                'total_time': sum(chunk_times),
                'avg_chunks_per_doc': statistics.mean(chunk_counts) if chunk_counts else 0,
                'total_chunks': sum(chunk_counts),
                'memory_delta_mb': chunker_memory['rss_mb'] - loader_memory['rss_mb']
            },
            'full_service': {
                'avg_time': statistics.mean(service_times) if service_times else 0,
                'total_time': sum(service_times),
                'avg_chunks_per_doc': statistics.mean(service_chunks) if service_chunks else 0,
                'docs_processed': len(service_times),
                'memory_delta_mb': end_memory['rss_mb'] - chunker_memory['rss_mb']
            }
        }
        
        print(f"TextLoader: {results['text_loader']['docs_loaded']} docs in {results['text_loader']['total_time']:.3f}s")
        print(f"DocumentChunker: {results['document_chunker']['total_chunks']} chunks in {results['document_chunker']['total_time']:.3f}s")
        print(f"Full Service: {results['full_service']['docs_processed']} docs in {results['full_service']['total_time']:.3f}s")
        
        return results
    
    def benchmark_embedding_service(self, sample_texts: List[str], batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict[str, Any]:
        """Benchmark EmbeddingService with different batch sizes."""
        print("\n=== BENCHMARKING EMBEDDING SERVICE ===")
        
        from src.embedding_service import EmbeddingService
        
        embedding_path = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        service = EmbeddingService(model_path=embedding_path)
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n--- Testing Batch Size: {batch_size} ---")
            
            # Limit texts to prevent memory issues
            test_texts = sample_texts[:min(len(sample_texts), batch_size * 10)]
            
            start_memory = self.get_memory_usage()
            start_time = time.time()
            
            try:
                embeddings = service.embed_texts(test_texts, batch_size=batch_size)
                
                embed_time = time.time() - start_time
                end_memory = self.get_memory_usage()
                
                batch_results[batch_size] = {
                    'texts_processed': len(test_texts),
                    'total_time': embed_time,
                    'texts_per_second': len(test_texts) / embed_time,
                    'time_per_text': embed_time / len(test_texts),
                    'embedding_shape': embeddings.shape if hasattr(embeddings, 'shape') else str(type(embeddings)),
                    'memory_delta_mb': end_memory['rss_mb'] - start_memory['rss_mb']
                }
                
                print(f"  Processed {len(test_texts)} texts in {embed_time:.3f}s ({len(test_texts)/embed_time:.1f} texts/sec)")
                
            except Exception as e:
                print(f"  ERROR with batch size {batch_size}: {e}")
                batch_results[batch_size] = {'error': str(e)}
        
        return batch_results
    
    def benchmark_vector_database(self, sample_chunks: List, sample_embeddings: List) -> Dict[str, Any]:
        """Benchmark VectorDatabase operations."""
        print("\n=== BENCHMARKING VECTOR DATABASE ===")
        
        from src.vector_database import VectorDatabase
        import uuid
        
        db_path = "data/benchmark_test.db"
        db = VectorDatabase(db_path)
        collection_id = "benchmark_test"
        
        # Clean up
        try:
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
                cursor.execute("DELETE FROM chunks WHERE collection_id = ?", (collection_id,))
                conn.commit()
        except:
            pass
        
        start_memory = self.get_memory_usage()
        
        # Benchmark document insertion
        print("\n--- Document Insertion ---")
        doc_insert_times = []
        
        for i in range(min(len(sample_chunks), 20)):  # Test first 20
            doc_id = str(uuid.uuid4())
            metadata = {'test': True, 'index': i}
            
            start_time = time.time()
            try:
                db.insert_document(doc_id, f"test_doc_{i}.txt", metadata, collection_id)
                insert_time = time.time() - start_time
                doc_insert_times.append(insert_time)
            except Exception as e:
                print(f"ERROR inserting document {i}: {e}")
        
        # Benchmark chunk insertion  
        print("\n--- Chunk Insertion ---")
        chunk_insert_times = []
        
        for i, (chunk, embedding) in enumerate(zip(sample_chunks[:20], sample_embeddings[:20])):
            start_time = time.time()
            try:
                db.insert_chunk(chunk, embedding, collection_id)
                insert_time = time.time() - start_time
                chunk_insert_times.append(insert_time)
            except Exception as e:
                print(f"ERROR inserting chunk {i}: {e}")
        
        # Benchmark search operations
        print("\n--- Search Operations ---")
        search_times = []
        
        if sample_embeddings:
            for i in range(min(5, len(sample_embeddings))):  # Test 5 searches
                query_embedding = sample_embeddings[i]
                
                start_time = time.time()
                try:
                    results = db.search_similar(query_embedding, collection_id=collection_id, limit=5)
                    search_time = time.time() - start_time
                    search_times.append(search_time)
                except Exception as e:
                    print(f"ERROR searching with embedding {i}: {e}")
        
        end_memory = self.get_memory_usage()
        
        results = {
            'document_insertion': {
                'count': len(doc_insert_times),
                'avg_time': statistics.mean(doc_insert_times) if doc_insert_times else 0,
                'total_time': sum(doc_insert_times),
                'docs_per_second': len(doc_insert_times) / sum(doc_insert_times) if doc_insert_times else 0
            },
            'chunk_insertion': {
                'count': len(chunk_insert_times), 
                'avg_time': statistics.mean(chunk_insert_times) if chunk_insert_times else 0,
                'total_time': sum(chunk_insert_times),
                'chunks_per_second': len(chunk_insert_times) / sum(chunk_insert_times) if chunk_insert_times else 0
            },
            'search_operations': {
                'count': len(search_times),
                'avg_time': statistics.mean(search_times) if search_times else 0,
                'total_time': sum(search_times),
                'searches_per_second': len(search_times) / sum(search_times) if search_times else 0
            },
            'memory_delta_mb': end_memory['rss_mb'] - start_memory['rss_mb']
        }
        
        print(f"Document Insertion: {results['document_insertion']['docs_per_second']:.1f} docs/sec")
        print(f"Chunk Insertion: {results['chunk_insertion']['chunks_per_second']:.1f} chunks/sec") 
        print(f"Search: {results['search_operations']['searches_per_second']:.1f} searches/sec")
        
        return results
    
    async def benchmark_corpus_manager(self, worker_configs: List[Dict[str, int]] = None) -> Dict[str, Any]:
        """Benchmark CorpusManager with different configurations."""
        print("\n=== BENCHMARKING CORPUS MANAGER ===")
        
        from src.corpus_manager import CorpusManager
        
        if worker_configs is None:
            worker_configs = [
                {'max_workers': 1, 'batch_size': 4},
                {'max_workers': 1, 'batch_size': 8},
                {'max_workers': 1, 'batch_size': 16}
            ]
        
        embedding_path = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        db_path = "data/benchmark_test.db"
        
        config_results = {}
        
        for config in worker_configs:
            config_name = f"w{config['max_workers']}_b{config['batch_size']}"
            print(f"\n--- Testing Configuration: {config_name} ---")
            
            collection_id = f"benchmark_{config_name}"
            
            # Clean up
            from src.vector_database import VectorDatabase
            db = VectorDatabase(db_path)
            try:
                with db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
                    cursor.execute("DELETE FROM chunks WHERE collection_id = ?", (collection_id,))
                    conn.commit()
            except:
                pass
            
            manager = CorpusManager(
                db_path=db_path,
                embedding_model_path=embedding_path,
                max_workers=config['max_workers'],
                batch_size=config['batch_size'],
                checkpoint_interval=10
            )
            
            start_memory = self.get_memory_usage()
            start_time = time.time()
            
            try:
                # Test with first 20 files only for speed
                files = sorted(self.test_dir.glob('*.txt'))[:20]
                
                result = await manager.ingest_directory(
                    path=self.test_dir,
                    collection_id=collection_id,
                    pattern='*.txt',
                    dry_run=False,
                    resume=False
                )
                
                total_time = time.time() - start_time
                end_memory = self.get_memory_usage()
                
                config_results[config_name] = {
                    'config': config,
                    'files_processed': result.files_processed,
                    'chunks_created': result.chunks_created,
                    'files_failed': result.files_failed,
                    'total_time': total_time,
                    'files_per_second': result.files_processed / total_time if total_time > 0 else 0,
                    'chunks_per_second': result.chunks_created / total_time if total_time > 0 else 0,
                    'memory_delta_mb': end_memory['rss_mb'] - start_memory['rss_mb']
                }
                
                print(f"  Processed {result.files_processed} files in {total_time:.2f}s ({result.files_processed/total_time:.2f} files/sec)")
                
            except Exception as e:
                print(f"  ERROR with config {config_name}: {e}")
                config_results[config_name] = {'error': str(e)}
        
        return config_results
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("🚀 STARTING COMPREHENSIVE PIPELINE BENCHMARK")
        print(f"Test corpus: {self.test_dir} ({len(list(self.test_dir.glob('*.txt')))} files)")
        
        # Get sample files
        files = sorted(self.test_dir.glob('*.txt'))
        sample_files = files[:20]  # Use first 20 for detailed testing
        
        # Benchmark 1: File Loading
        self.results['file_loading'] = self.benchmark_file_loading(sample_files)
        
        # Benchmark 2: Document Ingestion
        self.results['document_ingestion'] = self.benchmark_document_ingestion(sample_files)
        
        # Prepare sample data for other benchmarks
        from src.document_ingestion import DocumentIngestionService
        service = DocumentIngestionService()
        sample_chunks = []
        sample_texts = []
        
        for file_path in sample_files[:10]:  # Use fewer files for embedding tests
            try:
                chunks = service.ingest_document(str(file_path))
                sample_chunks.extend(chunks)
                sample_texts.extend([chunk.content for chunk in chunks])
            except Exception as e:
                print(f"ERROR preparing sample data from {file_path}: {e}")
        
        # Benchmark 3: Embedding Service
        self.results['embedding_service'] = self.benchmark_embedding_service(sample_texts[:50])  # Limit to 50 texts
        
        # Generate sample embeddings for database testing
        from src.embedding_service import EmbeddingService
        embedding_path = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        embed_service = EmbeddingService(model_path=embedding_path)
        sample_embeddings = embed_service.embed_texts(sample_texts[:20])  # Generate 20 embeddings
        
        # Benchmark 4: Vector Database
        self.results['vector_database'] = self.benchmark_vector_database(sample_chunks[:20], sample_embeddings)
        
        # Benchmark 5: Corpus Manager
        self.results['corpus_manager'] = await self.benchmark_corpus_manager()
        
        # Calculate overall metrics
        self.calculate_overall_metrics()
        
        return self.results
    
    def calculate_overall_metrics(self):
        """Calculate pipeline-wide performance metrics."""
        print("\n=== CALCULATING OVERALL METRICS ===")
        
        # Extract key metrics
        overall = {}
        
        if 'document_ingestion' in self.results:
            service_data = self.results['document_ingestion']['full_service']
            overall['document_processing_rate'] = 1 / service_data['avg_time'] if service_data['avg_time'] > 0 else 0
        
        if 'embedding_service' in self.results:
            # Find best batch size performance
            best_batch = None
            best_rate = 0
            for batch_size, data in self.results['embedding_service'].items():
                if isinstance(data, dict) and 'texts_per_second' in data:
                    if data['texts_per_second'] > best_rate:
                        best_rate = data['texts_per_second']
                        best_batch = batch_size
            overall['best_embedding_batch_size'] = best_batch
            overall['best_embedding_rate'] = best_rate
        
        if 'vector_database' in self.results:
            overall['chunk_insert_rate'] = self.results['vector_database']['chunk_insertion']['chunks_per_second']
            overall['search_rate'] = self.results['vector_database']['search_operations']['searches_per_second']
        
        if 'corpus_manager' in self.results:
            # Find best configuration
            best_config = None
            best_rate = 0
            for config_name, data in self.results['corpus_manager'].items():
                if isinstance(data, dict) and 'files_per_second' in data:
                    if data['files_per_second'] > best_rate:
                        best_rate = data['files_per_second']
                        best_config = config_name
            overall['best_pipeline_config'] = best_config
            overall['best_pipeline_rate'] = best_rate
        
        self.results['overall_metrics'] = overall
        
        print("Overall Performance Summary:")
        for metric, value in overall.items():
            print(f"  {metric}: {value}")
    
    def save_results(self, filename: str = "pipeline_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📊 Benchmark results saved to: {filename}")

async def main():
    """Run the complete benchmark suite."""
    benchmarker = PipelineBenchmarker("data/test_100")
    
    try:
        results = await benchmarker.run_full_benchmark()
        benchmarker.save_results()
        
        print("\n🎉 BENCHMARK COMPLETE!")
        print("Check pipeline_benchmark_results.json for detailed results.")
        
    except Exception as e:
        print(f"❌ BENCHMARK FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())