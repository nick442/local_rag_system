#!/usr/bin/env python3
"""Test script to verify sqlite-vec extension fix"""

import sys
import logging
import tempfile
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add src to path and fix relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the components directly to test sqlite-vec
import sqlite3
import sqlite_vec
import numpy as np
import tempfile

def test_sqlite_vec_fix():
    """Test that sqlite-vec extension loads properly and vector search works"""
    
    print("🧪 Testing sqlite-vec extension fix...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Test direct sqlite-vec loading
        print(f"Creating SQLite connection to: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Load sqlite-vec using Python package
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            print("✅ sqlite-vec extension loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load sqlite-vec: {e}")
            return False
        
        # Test basic connection and extension loading
        cursor = conn.cursor()
        
        # Test if sqlite-vec functions are available
        try:
            cursor.execute("SELECT vec_version()")
            version = cursor.fetchone()[0]
            print(f"✅ sqlite-vec version: {version}")
        except Exception as e:
            print(f"❌ vec_version() failed: {e}")
            return False
        
        # Test creating vector table
        try:
            cursor.execute("""
                CREATE VIRTUAL TABLE test_vec USING vec0(
                    embedding float[384]
                )
            """)
            print("✅ Vector virtual table creation successful")
        except Exception as e:
            print(f"❌ Vector table creation failed: {e}")
            return False
        
        # Test inserting vector data
        try:
            test_embedding = np.random.rand(384).astype(np.float32)
            # Convert to JSON string format as shown in the documentation
            embedding_json = f"[{','.join(map(str, test_embedding.tolist()))}]"
            cursor.execute("""
                INSERT INTO test_vec(rowid, embedding) VALUES (?, ?)
            """, (1, embedding_json))
            conn.commit()
            print("✅ Vector insertion successful")
        except Exception as e:
            print(f"❌ Vector insertion failed: {e}")
            return False
        
        # Test vector similarity search
        try:
            query_embedding = np.random.rand(384).astype(np.float32)
            query_json = f"[{','.join(map(str, query_embedding.tolist()))}]"
            cursor.execute("""
                SELECT rowid, distance
                FROM test_vec
                WHERE embedding match ?
                ORDER BY distance
                LIMIT 1
            """, (query_json,))
            
            result = cursor.fetchone()
            if result:
                print(f"✅ Vector similarity search successful! ID: {result[0]}, Distance: {result[1]:.6f}")
            else:
                print("❌ No results from vector search")
                return False
                
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return False
            
        conn.close()
        
        print("🎉 All tests passed! sqlite-vec extension is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)

if __name__ == "__main__":
    success = test_sqlite_vec_fix()
    sys.exit(0 if success else 1)