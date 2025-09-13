#!/usr/bin/env python3
"""
Convert JSONL corpus files to text format for ingestion.

Converts the FIQA and SciFact datasets from JSONL format to individual
text files that can be ingested by the RAG system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import html

def clean_text(text: str) -> str:
    """Clean and normalize text for ingestion."""
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove excessive punctuation
    text = text.replace('..', '.')
    text = text.replace('???', '?')
    text = text.replace('!!!', '!')
    
    return text.strip()

def convert_fiqa_corpus(input_file: Path, output_dir: Path) -> int:
    """Convert FIQA JSONL corpus to individual text files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract fields
                doc_id = data.get('_id', f'fiqa_{line_num}')
                title = clean_text(data.get('title', ''))
                text = clean_text(data.get('text', ''))
                
                if not text:
                    continue
                
                # Combine title and text
                content = ""
                if title:
                    content = f"{title}\n\n{text}"
                else:
                    content = text
                
                # Write to file
                output_file = output_dir / f"{doc_id}.txt"
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(content)
                
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} FIQA documents...")
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_num}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return processed_count

def convert_scifact_corpus(input_file: Path, output_dir: Path) -> int:
    """Convert SciFact JSONL corpus to individual text files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract fields
                doc_id = data.get('_id', f'scifact_{line_num}')
                title = clean_text(data.get('title', ''))
                text = clean_text(data.get('text', ''))
                
                if not text:
                    continue
                
                # Combine title and text
                content = ""
                if title:
                    content = f"{title}\n\n{text}"
                else:
                    content = text
                
                # Write to file
                output_file = output_dir / f"{doc_id}.txt"
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(content)
                
                processed_count += 1
                
                if processed_count % 500 == 0:
                    print(f"Processed {processed_count} SciFact documents...")
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_num}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return processed_count

def main():
    """Convert both corpus datasets."""
    base_dir = Path("/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/Opus-Experiments")
    
    # FIQA conversion
    print("Converting FIQA corpus...")
    fiqa_input = base_dir / "corpus" / "technical" / "fiqa" / "corpus.jsonl"
    fiqa_output = base_dir / "corpus" / "processed" / "fiqa_technical"
    
    if fiqa_input.exists():
        fiqa_count = convert_fiqa_corpus(fiqa_input, fiqa_output)
        print(f"✓ FIQA conversion complete: {fiqa_count} documents")
    else:
        print(f"✗ FIQA input file not found: {fiqa_input}")
    
    # SciFact conversion
    print("\nConverting SciFact corpus...")
    scifact_input = base_dir / "corpus" / "narrative" / "scifact" / "corpus.jsonl"
    scifact_output = base_dir / "corpus" / "processed" / "scifact_scientific"
    
    if scifact_input.exists():
        scifact_count = convert_scifact_corpus(scifact_input, scifact_output)
        print(f"✓ SciFact conversion complete: {scifact_count} documents")
    else:
        print(f"✗ SciFact input file not found: {scifact_input}")
    
    print(f"\nConversion complete!")
    print(f"Processed directories:")
    print(f"- FIQA: {fiqa_output}")
    print(f"- SciFact: {scifact_output}")

if __name__ == "__main__":
    main()