#!/usr/bin/env python3
"""
Create a subset of queries whose IDs appear in a BEIR qrels TSV.

Usage:
  python tools/make_query_subset_from_qrels.py \
    --queries test_data/beir_fiqa_queries.json \
    --qrels test_data/beir_fiqa_qrels_test.tsv \
    --out test_data/fiqa_queries_subset_test_100.json \
    --n 100 --seed 42
"""

import json
import random
from pathlib import Path
import argparse

def load_qids_from_qrels(path: Path):
    qids = set()
    with open(path, 'r') as f:
        f.readline()
        for ln in f:
            parts = ln.strip().split('\t')
            if len(parts) >= 1:
                qids.add(str(parts[0]))
    return qids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True)
    ap.add_argument('--qrels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    qrels_qids = load_qids_from_qrels(Path(args.qrels))
    data = json.loads(Path(args.queries).read_text())
    queries = [q for q in data.get('queries', []) if str(q.get('id')) in qrels_qids]
    random.seed(args.seed)
    if len(queries) > args.n:
        sub = random.sample(queries, args.n)
    else:
        sub = queries
    out_data = { 'metadata': data.get('metadata', {}), 'queries': sub }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_data, indent=2))
    print(f"Selected {len(sub)}/{len(queries)} queries with qrels from {args.queries} -> {args.out}")

if __name__ == '__main__':
    main()

