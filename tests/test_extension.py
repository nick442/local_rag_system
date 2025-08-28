#!/usr/bin/env python3
"""Test script to debug sqlite-vec extension loading."""

import sqlite3
import os
import sys

print(f"Python version: {sys.version}")
print(f"SQLite version: {sqlite3.sqlite_version}")
print(f"Current working directory: {os.getcwd()}")

try:
    conn = sqlite3.connect(':memory:')
    print("SQLite connection created successfully")
    
    # Check if extension loading is supported
    try:
        conn.enable_load_extension(True)
        print("Extension loading enabled successfully")
    except Exception as e:
        print(f"Failed to enable extension loading: {e}")
        sys.exit(1)
    
    # Try to load the extension
    ext_path = os.path.abspath('vec0.dylib')
    print(f"Extension path: {ext_path}")
    print(f"Extension exists: {os.path.exists(ext_path)}")
    
    if os.path.exists(ext_path):
        print(f"Extension file size: {os.path.getsize(ext_path)} bytes")
        # Check permissions
        print(f"Extension is readable: {os.access(ext_path, os.R_OK)}")
        print(f"Extension is executable: {os.access(ext_path, os.X_OK)}")
    
    try:
        conn.load_extension(ext_path)
        print("SUCCESS: sqlite-vec extension loaded!")
        
        # Test the extension
        cursor = conn.cursor()
        cursor.execute('SELECT vec_version()')
        version = cursor.fetchone()
        print(f"sqlite-vec version: {version[0]}")
        
    except Exception as e:
        print(f"FAILED to load extension: {e}")
        print(f"Error type: {type(e).__name__}")
        # Try alternative methods
        try:
            conn.load_extension("vec0")
            print("SUCCESS: Loaded as 'vec0'")
        except Exception as e2:
            print(f"Also failed with 'vec0': {e2}")
            
        try:
            conn.load_extension("sqlite-vec")
            print("SUCCESS: Loaded as 'sqlite-vec'")
        except Exception as e3:
            print(f"Also failed with 'sqlite-vec': {e3}")

except Exception as e:
    print(f"Failed to create SQLite connection: {e}")
    sys.exit(1)

print("Test completed")