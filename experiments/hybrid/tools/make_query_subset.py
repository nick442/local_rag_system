#!/usr/bin/env python3
"""
Create a subset of queries from a test_data/beir_*_queries.json, preserving ids.

Usage:
  python tools/make_query_subset.py \
    --in test_data/beir_fiqa_queries.json \
    --out test_data/fiqa_queries_subset_100.json \
    --n 100 --seed 42
"""

import json
import random
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    data = json.loads(inp.read_text())
    queries = data.get('queries', [])
    random.seed(args.seed)
    if len(queries) > args.n:
        sub = random.sample(queries, args.n)
    else:
        sub = queries
    out_data = {
        'metadata': data.get('metadata', {}),
        'queries': sub
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out_data, indent=2))
    print(f"Wrote {len(sub)} queries to {outp}")

if __name__ == '__main__':
    main()

