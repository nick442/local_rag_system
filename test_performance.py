#!/usr/bin/env python3
"""
Performance testing script for corpus ingestion with extensive logging.
"""

import time
import logging
import sys
from pathlib import Path
from src.corpus_manager import CorpusManager
from src.vector_database import VectorDatabase

# Configure extensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('performance_test.log')
    ]
)

def test_ingestion_performance():
    """Test corpus ingestion performance with detailed measurement."""
    
    print("=== CORPUS INGESTION PERFORMANCE TEST ===")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test parameters
    test_dir = "data/test_200"
    collection_id = "test_200_performance"
    db_path = "data/rag_vectors.db"
    
    # Initialize components
    print(f"\n1. Initializing components...")
    start_init = time.time()
    
    # Use default embedding model path
    embedding_path = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    
    manager = CorpusManager(
        db_path=db_path,
        embedding_model_path=embedding_path,
        max_workers=1,  # Single worker for consistent measurement
        batch_size=8,   # Small batch for detailed tracking
        checkpoint_interval=5  # Checkpoint every 5 files
    )
    
    db = VectorDatabase(db_path)
    
    init_time = time.time() - start_init
    print(f"   Initialization took: {init_time:.2f} seconds")
    
    # Check test directory
    print(f"\n2. Scanning test directory...")
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"ERROR: Test directory {test_dir} does not exist!")
        return
    
    files = list(test_path.glob("*.txt"))
    print(f"   Found {len(files)} files to process")
    
    if len(files) == 0:
        print("ERROR: No .txt files found in test directory!")
        return
    
    # Clear any existing documents in test collection
    print(f"\n3. Cleaning up previous test data...")
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
            cursor.execute("DELETE FROM chunks WHERE collection_id = ?", (collection_id,))
            deleted_docs = cursor.rowcount
            conn.commit()
            print(f"   Cleared {deleted_docs} existing documents from collection")
    except Exception as e:
        print(f"   Cleanup warning: {e}")
    
    # Start ingestion test
    print(f"\n4. Starting ingestion test...")
    print(f"   Collection: {collection_id}")
    print(f"   Files: {len(files)}")
    print(f"   Workers: 1")
    print(f"   Batch size: 8")
    
    start_time = time.time()
    last_checkpoint_time = start_time
    
    try:
        # Use the ingest_directory method for consistent testing
        import asyncio
        result = asyncio.run(manager.ingest_directory(
            path=test_path,
            collection_id=collection_id,
            pattern="*.txt",
            dry_run=False,
            resume=False
        ))
        
        total_time = time.time() - start_time
        
        print(f"\n5. Ingestion completed!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Files processed: {result.files_processed}")
        print(f"   Chunks created: {result.chunks_created}")
        print(f"   Files failed: {result.files_failed}")
        print(f"   Average time per file: {total_time / result.files_processed:.3f} seconds")
        print(f"   Processing rate: {result.files_processed / total_time:.2f} files/second")
        
        # Verify database contents
        print(f"\n6. Verifying database contents...")
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE collection_id = ?", (collection_id,))
            doc_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE collection_id = ?", (collection_id,))
            chunk_count = cursor.fetchone()[0]
            
            print(f"   Documents in database: {doc_count}")
            print(f"   Chunks in database: {chunk_count}")
            print(f"   Match expected: {doc_count == result.files_processed}")
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\nERROR: Ingestion failed after {total_time:.2f} seconds")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== TEST COMPLETED ===")
    print(f"Total test time: {time.time() - start_time:.2f} seconds")
    print(f"Log file: performance_test.log")

if __name__ == "__main__":
    test_ingestion_performance()