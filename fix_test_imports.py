#!/usr/bin/env python3
"""Fix import paths in moved test files"""

import os
from pathlib import Path

tests_dir = Path("tests")
old_pattern = "sys.path.insert(0, str(Path(__file__).parent))"
new_pattern = "sys.path.insert(0, str(Path(__file__).parent.parent))"

for test_file in tests_dir.glob("test*.py"):
    print(f"Fixing {test_file}...")
    
    # Read file
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Replace the import path
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        
        # Write back
        with open(test_file, 'w') as f:
            f.write(content)
        
        print(f"  ✅ Fixed import path")
    else:
        print(f"  ℹ️  No import path to fix")

print("All test files processed!")