#!/usr/bin/env python3
"""
Augment experiment results with retrieval-quality metrics (NDCG@10, Recall@10).

Requirements:
- A results JSON produced by main.py experiment sweep (with metrics.retrieved_doc_ids present)
- test_data/beir_{dataset}_qrels_test.tsv (tab-separated: query-id, corpus-id, score)
- test_data/beir_{dataset}_queries.json (with entries {id, query}) used during the sweep
- A SQLite DB path with documents table mapping doc_uuid -> source_path, where
  source_path basename (without extension) equals BEIR doc id.

Usage example:
  python tools/augment_results_with_quality.py \
    --dataset fiqa \
    --results experiments/hybrid/results/fiqa_alpha_sweep.json \
    --db data/rag_vectors.db \
    --out experiments/hybrid/results/fiqa_alpha_sweep.quality.json
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

def load_qrels_tsv(path: Path) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    with open(path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            qid, docid, score = parts[0], parts[1], float(parts[2])
            qrels.setdefault(qid, {})[docid] = score
    return qrels

def build_doc_uuid_to_beir_id(db_path: Path) -> Dict[str, str]:
    """Map internal doc_uuid -> BEIR document id extracted from source_path basename."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    mapping: Dict[str, str] = {}
    for row in cur.execute("select doc_id, source_path from documents"):
        sp = row["source_path"] or ""
        beir_id = Path(sp).stem
        mapping[row["doc_id"]] = beir_id
    conn.close()
    return mapping

def ndcg_at_k(retrieved: List[str], rels: Dict[str, float], k: int = 10) -> float:
    import math
    gains = [rels.get(doc, 0.0) for doc in retrieved[:k]]
    dcg = sum(g / math.log2(i+2) for i, g in enumerate(gains))
    ideal = sorted(rels.values(), reverse=True)
    idcg = sum(g / math.log2(i+2) for i, g in enumerate(ideal[:k]))
    return (dcg / idcg) if idcg > 0 else 0.0

def recall_at_k(retrieved: List[str], rels: Dict[str, float], k: int = 10) -> float:
    if not rels:
        return 0.0
    topk = set(retrieved[:k])
    relset = {d for d, s in rels.items() if s > 0}
    return len(topk & relset) / len(relset) if relset else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['fiqa','scifact'])
    ap.add_argument('--results', required=True)
    ap.add_argument('--db', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    results_path = Path(args.results)
    out_path = Path(args.out)
    dataset = args.dataset
    qrels_path = Path(f"test_data/beir_{dataset}_qrels_test.tsv")

    if not qrels_path.exists():
        raise SystemExit(f"Missing qrels file: {qrels_path}")

    with open(results_path, 'r') as f:
        data = json.load(f)

    qrels = load_qrels_tsv(qrels_path)
    uuid_to_beir = build_doc_uuid_to_beir_id(Path(args.db))

    enriched = data
    for r in enriched.get('results', []):
        metrics = r.get('metrics', {})
        qid = str(metrics.get('query_id') or r.get('query_id') or '')
        beir_rels = qrels.get(qid)
        docs_uuid = metrics.get('retrieved_doc_ids') or []
        docs_beir = [uuid_to_beir.get(did, '') for did in docs_uuid]
        docs_beir = [d for d in docs_beir if d]
        if beir_rels and docs_beir:
            ndcg = ndcg_at_k(docs_beir, beir_rels, k=10)
            rec = recall_at_k(docs_beir, beir_rels, k=10)
            metrics['ndcg@10'] = ndcg
            metrics['recall@10'] = rec
        r['metrics'] = metrics

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(enriched, f, indent=2)
    print(f"Wrote enriched results to: {out_path}")

if __name__ == '__main__':
    main()

