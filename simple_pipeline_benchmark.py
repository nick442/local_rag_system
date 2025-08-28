#!/usr/bin/env python3
"""
Simple pipeline timing benchmark focused on measuring component performance.
"""

import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
import json

def benchmark_components():
    """Run focused timing benchmark of each pipeline component."""
    
    print("🚀 SIMPLE PIPELINE COMPONENT TIMING BENCHMARK")
    print("=" * 60)
    
    test_dir = Path("data/test_100")
    files = sorted(test_dir.glob("*.txt"))[:20]  # First 20 files
    
    results = {}
    
    # 1. File Loading Benchmark
    print("\n📁 BENCHMARKING FILE LOADING")
    load_times = []
    total_chars = 0
    
    for file_path in files:
        start = time.time()
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        load_time = time.time() - start
        
        load_times.append(load_time)
        total_chars += len(content)
    
    results['file_loading'] = {
        'files': len(files),
        'total_time': sum(load_times),
        'avg_time_per_file': statistics.mean(load_times),
        'chars_per_second': total_chars / sum(load_times),
        'total_chars': total_chars
    }
    
    print(f"  Files loaded: {len(files)}")
    print(f"  Total time: {sum(load_times):.4f}s")
    print(f"  Avg time per file: {statistics.mean(load_times):.4f}s")
    print(f"  Speed: {total_chars / sum(load_times):.0f} chars/sec")
    
    # 2. Document Ingestion Benchmark
    print("\n📄 BENCHMARKING DOCUMENT INGESTION")
    from src.document_ingestion import DocumentIngestionService
    
    service = DocumentIngestionService(chunk_size=512, chunk_overlap=128)
    ingestion_times = []
    total_chunks = 0
    all_chunks = []
    
    for file_path in files:
        start = time.time()
        chunks = service.ingest_document(str(file_path))
        ingestion_time = time.time() - start
        
        ingestion_times.append(ingestion_time)
        total_chunks += len(chunks)
        all_chunks.extend(chunks)
    
    results['document_ingestion'] = {
        'files': len(files),
        'total_time': sum(ingestion_times),
        'avg_time_per_file': statistics.mean(ingestion_times),
        'files_per_second': len(files) / sum(ingestion_times),
        'total_chunks': total_chunks,
        'avg_chunks_per_file': total_chunks / len(files)
    }
    
    print(f"  Files processed: {len(files)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total time: {sum(ingestion_times):.4f}s")
    print(f"  Speed: {len(files) / sum(ingestion_times):.2f} files/sec")
    print(f"  Avg chunks per file: {total_chunks / len(files):.1f}")
    
    # 3. Embedding Generation Benchmark
    print("\n🧠 BENCHMARKING EMBEDDING GENERATION")
    from src.embedding_service import EmbeddingService
    
    embedding_path = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    embed_service = EmbeddingService(model_path=embedding_path, batch_size=8)
    
    # Test with chunk texts from document ingestion
    chunk_texts = [chunk.content for chunk in all_chunks[:50]]  # First 50 chunks
    
    print(f"  Testing with {len(chunk_texts)} text chunks...")
    
    start = time.time()
    embeddings = embed_service.embed_texts(chunk_texts, show_progress=False)
    embed_time = time.time() - start
    
    results['embedding_generation'] = {
        'texts_processed': len(chunk_texts),
        'total_time': embed_time,
        'texts_per_second': len(chunk_texts) / embed_time,
        'time_per_text': embed_time / len(chunk_texts),
        'embedding_dimension': len(embeddings[0]) if embeddings else 0
    }
    
    print(f"  Texts processed: {len(chunk_texts)}")
    print(f"  Total time: {embed_time:.4f}s")
    print(f"  Speed: {len(chunk_texts) / embed_time:.2f} texts/sec")
    print(f"  Time per text: {embed_time / len(chunk_texts):.4f}s")
    
    # 4. Database Operations Benchmark
    print("\n🗄️  BENCHMARKING DATABASE OPERATIONS")
    from src.vector_database import VectorDatabase
    import uuid
    
    # Use a test database
    db_path = "data/simple_benchmark_test.db"
    db = VectorDatabase(db_path)
    collection_id = "benchmark_test"
    
    # Clean up first
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            # Just try to delete without collection_id if that column doesn't exist
            try:
                cursor.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
                cursor.execute("DELETE FROM chunks WHERE collection_id = ?", (collection_id,))
            except:
                cursor.execute("DELETE FROM documents")
                cursor.execute("DELETE FROM chunks") 
            conn.commit()
    except:
        pass
    
    # Test document insertion
    doc_insert_times = []
    test_chunks = all_chunks[:10]  # First 10 chunks
    test_embeddings = embeddings[:10] if len(embeddings) >= 10 else embeddings
    
    for i, chunk in enumerate(test_chunks):
        doc_id = str(uuid.uuid4())
        metadata = {'test': True, 'index': i}
        
        start = time.time()
        try:
            # Try with collection_id first, fall back without it
            try:
                db.insert_document(doc_id, f"test_doc_{i}.txt", metadata, collection_id)
            except:
                db.insert_document(doc_id, f"test_doc_{i}.txt", metadata)
            insert_time = time.time() - start
            doc_insert_times.append(insert_time)
        except Exception as e:
            print(f"    Document insert failed: {e}")
            break
    
    # Test chunk insertion
    chunk_insert_times = []
    
    for i, (chunk, embedding) in enumerate(zip(test_chunks, test_embeddings)):
        start = time.time()
        try:
            # Try with collection_id first, fall back without it
            try:
                db.insert_chunk(chunk, embedding, collection_id)
            except:
                db.insert_chunk(chunk, embedding)
            insert_time = time.time() - start
            chunk_insert_times.append(insert_time)
        except Exception as e:
            print(f"    Chunk insert failed: {e}")
            break
    
    results['database_operations'] = {
        'document_inserts': len(doc_insert_times),
        'doc_insert_total_time': sum(doc_insert_times) if doc_insert_times else 0,
        'docs_per_second': len(doc_insert_times) / sum(doc_insert_times) if doc_insert_times else 0,
        'chunk_inserts': len(chunk_insert_times),
        'chunk_insert_total_time': sum(chunk_insert_times) if chunk_insert_times else 0,
        'chunks_per_second': len(chunk_insert_times) / sum(chunk_insert_times) if chunk_insert_times else 0
    }
    
    print(f"  Document inserts: {len(doc_insert_times)}")
    if doc_insert_times:
        print(f"  Doc insert speed: {len(doc_insert_times) / sum(doc_insert_times):.2f} docs/sec")
    
    print(f"  Chunk inserts: {len(chunk_insert_times)}")  
    if chunk_insert_times:
        print(f"  Chunk insert speed: {len(chunk_insert_times) / sum(chunk_insert_times):.2f} chunks/sec")
    
    # 5. End-to-End Pipeline Timing
    print("\n⚡ BENCHMARKING END-TO-END PIPELINE")
    from src.corpus_manager import CorpusManager
    
    # Use smaller test for timing
    small_test_files = files[:5]  # Just 5 files for quick test
    
    embedding_path = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    manager = CorpusManager(
        db_path="data/simple_benchmark_test.db",
        embedding_model_path=embedding_path,
        max_workers=1,
        batch_size=4,
        checkpoint_interval=10
    )
    
    # Process individual files to avoid directory scanning overhead
    pipeline_times = []
    pipeline_chunks = []
    pipeline_successes = 0
    
    for file_path in small_test_files:
        start = time.time()
        try:
            success, stats = manager.process_single_document(file_path, "pipeline_test")
            pipeline_time = time.time() - start
            
            if success:
                pipeline_times.append(pipeline_time)
                pipeline_chunks.append(stats.get('chunks_created', 0))
                pipeline_successes += 1
            else:
                print(f"    Pipeline failed for {file_path.name}: {stats.get('error', 'unknown')}")
                
        except Exception as e:
            pipeline_time = time.time() - start
            print(f"    Pipeline error for {file_path.name}: {e}")
    
    results['end_to_end_pipeline'] = {
        'files_tested': len(small_test_files),
        'files_succeeded': pipeline_successes,
        'total_time': sum(pipeline_times) if pipeline_times else 0,
        'avg_time_per_file': statistics.mean(pipeline_times) if pipeline_times else 0,
        'files_per_second': len(pipeline_times) / sum(pipeline_times) if pipeline_times else 0,
        'total_chunks': sum(pipeline_chunks),
        'avg_chunks_per_file': statistics.mean(pipeline_chunks) if pipeline_chunks else 0
    }
    
    print(f"  Files tested: {len(small_test_files)}")
    print(f"  Files succeeded: {pipeline_successes}")
    if pipeline_times:
        print(f"  Total time: {sum(pipeline_times):.4f}s")
        print(f"  Speed: {len(pipeline_times) / sum(pipeline_times):.2f} files/sec")
        print(f"  Avg chunks per file: {statistics.mean(pipeline_chunks):.1f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if results.get('file_loading'):
        print(f"File Loading:        {results['file_loading']['chars_per_second']:.0f} chars/sec")
    
    if results.get('document_ingestion'):
        print(f"Document Ingestion:  {results['document_ingestion']['files_per_second']:.2f} files/sec")
    
    if results.get('embedding_generation'):
        print(f"Embedding Generation: {results['embedding_generation']['texts_per_second']:.2f} texts/sec")
    
    if results.get('database_operations') and results['database_operations']['chunks_per_second'] > 0:
        print(f"Database Inserts:    {results['database_operations']['chunks_per_second']:.2f} chunks/sec")
    
    if results.get('end_to_end_pipeline') and results['end_to_end_pipeline']['files_per_second'] > 0:
        print(f"End-to-End Pipeline: {results['end_to_end_pipeline']['files_per_second']:.2f} files/sec")
    
    # Calculate projected time for 100 and 11k documents
    if results.get('end_to_end_pipeline') and results['end_to_end_pipeline']['files_per_second'] > 0:
        files_per_sec = results['end_to_end_pipeline']['files_per_second']
        time_100 = 100 / files_per_sec
        time_11k = 11000 / files_per_sec
        
        print(f"\n🔮 PROJECTIONS:")
        print(f"100 documents:   {time_100/60:.1f} minutes")
        print(f"11k documents:   {time_11k/3600:.1f} hours")
    
    # Save results
    with open("simple_benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: simple_benchmark_results.json")
    
    return results

if __name__ == "__main__":
    benchmark_components()