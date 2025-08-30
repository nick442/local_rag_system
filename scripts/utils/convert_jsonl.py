#!/usr/bin/env python3
"""Convert JSONL corpus to individual text files for ingestion."""

import json
import os
from pathlib import Path
import re

def clean_filename(text: str, max_length: int = 50) -> str:
    """Create a clean filename from text."""
    # Remove or replace problematic characters
    clean = re.sub(r'[^\w\s-]', '', text)
    clean = re.sub(r'[-\s]+', '-', clean)
    return clean[:max_length].strip('-')

def convert_jsonl_to_txt(jsonl_path: str, output_dir: str, limit: int = None):
    """Convert JSONL file to individual text files."""
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if limit and count >= limit:
                break
                
            try:
                doc = json.loads(line.strip())
                text = doc.get('text', '').strip()
                title = doc.get('title', f'document_{line_num}').strip()
                
                if not text or len(text) < 50:  # Skip very short documents
                    continue
                
                # Create filename
                if title and title != '':
                    filename = clean_filename(title)
                else:
                    filename = f"doc_{doc.get('id', line_num)}"
                
                # Ensure unique filename
                base_filename = filename
                counter = 1
                while (output_dir / f"{filename}.txt").exists():
                    filename = f"{base_filename}_{counter}"
                    counter += 1
                
                # Write content
                content = text
                if title and title.strip():
                    content = f"{title}\n\n{text}"
                
                file_path = output_dir / f"{filename}.txt"
                with open(file_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(content)
                
                count += 1
                if count % 1000 == 0:
                    print(f"Converted {count} documents...")
                    
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON at line {line_num}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Conversion complete: {count} documents converted to {output_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python convert_jsonl.py <input.jsonl> <output_dir> [limit]")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    output_dir = sys.argv[2] 
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    convert_jsonl_to_txt(jsonl_file, output_dir, limit)
