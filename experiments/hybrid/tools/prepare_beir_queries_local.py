#!/usr/bin/env python3
"""
Prepare BEIR queries from local corpus directory into test_data JSON files.

Sources:
- corpus/technical/fiqa/queries.jsonl
- corpus/narrative/scifact/queries.jsonl

Outputs:
- test_data/beir_fiqa_queries.json
- test_data/beir_scifact_queries.json
"""

import json
from pathlib import Path

def load_jsonl(fp: Path):
    items = []
    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({
                'id': str(obj.get('_id')),
                'query': obj.get('text', ''),
                'metadata': obj.get('metadata', {})
            })
    return items

def write_queries(queries, dest: Path, dataset: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, 'w') as f:
        json.dump({'metadata': {'dataset': dataset, 'source': 'local_corpus'}, 'queries': queries}, f, indent=2)

def main():
    root = Path(__file__).resolve().parents[3]
    fiqa_src = root / 'corpus' / 'technical' / 'fiqa' / 'queries.jsonl'
    scifact_src = root / 'corpus' / 'narrative' / 'scifact' / 'queries.jsonl'
    out_dir = root / 'test_data'

    if fiqa_src.exists():
        q = load_jsonl(fiqa_src)
        write_queries(q, out_dir / 'beir_fiqa_queries.json', 'fiqa')
        print(f"Wrote {len(q)} FiQA queries to {out_dir / 'beir_fiqa_queries.json'}")
    else:
        print(f"Missing: {fiqa_src}")

    if scifact_src.exists():
        q = load_jsonl(scifact_src)
        write_queries(q, out_dir / 'beir_scifact_queries.json', 'scifact')
        print(f"Wrote {len(q)} SciFact queries to {out_dir / 'beir_scifact_queries.json'}")
    else:
        print(f"Missing: {scifact_src}")

if __name__ == '__main__':
    main()
