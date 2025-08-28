#!/usr/bin/env python3
"""
Isolated testing of DocumentIngestionService to identify problematic documents.
"""

import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_document_ingestion_isolation():
    """Test document ingestion in complete isolation."""
    print("=== DOCUMENT INGESTION ISOLATION TEST ===")
    
    # Test specific files from the problematic sequence
    test_files = [
        "data/test_200/01-2015-10html.txt",  # File #8 - last successful
        "data/test_200/01.txt",              # File #9 - suspected problematic
        "data/test_200/0100safeshtml.txt",   # File #10 - next in sequence
        "data/test_200/013132296296.txt",    # File #11 - smaller file
    ]
    
    from src.document_ingestion import DocumentIngestionService
    
    print(f"\n1. Testing DocumentIngestionService in isolation...")
    service = DocumentIngestionService()
    
    for i, file_path in enumerate(test_files, 1):
        file_name = Path(file_path).name
        print(f"\n--- Test {i}: {file_name} ---")
        
        if not Path(file_path).exists():
            print(f"ERROR: File does not exist: {file_path}")
            continue
            
        file_size = Path(file_path).stat().st_size
        print(f"File size: {file_size} bytes")
        
        try:
            start_time = time.time()
            
            # Test just the document loading
            print("Loading document...")
            chunks = service.ingest_document(file_path)
            
            load_time = time.time() - start_time
            
            print(f"SUCCESS: Loaded {len(chunks)} chunks in {load_time:.3f}s")
            
            # Print chunk details
            for j, chunk in enumerate(chunks):
                print(f"  Chunk {j}: {len(chunk.content)} chars, {chunk.token_count} tokens")
                
        except Exception as e:
            error_time = time.time() - start_time
            print(f"ERROR after {error_time:.3f}s: {str(e)}")
            import traceback
            traceback.print_exc()
            
        print(f"Memory check - continuing...")

def test_file_reading_only():
    """Test just file reading without any processing."""
    print("\n=== RAW FILE READING TEST ===")
    
    test_files = [
        "data/test_200/01-2015-10html.txt",
        "data/test_200/01.txt", 
        "data/test_200/0100safeshtml.txt",
    ]
    
    for file_path in test_files:
        file_name = Path(file_path).name
        print(f"\n--- Reading {file_name} ---")
        
        try:
            start_time = time.time()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            read_time = time.time() - start_time
            print(f"SUCCESS: Read {len(content)} characters in {read_time:.3f}s")
            print(f"First 100 chars: {repr(content[:100])}")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_file_reading_only()
    test_document_ingestion_isolation()