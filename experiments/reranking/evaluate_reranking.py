#!/usr/bin/env python3
"""
Evaluate reranking results against BEIR-style qrels.

Inputs:
- Results JSON from run_reranking_experiment.py (contains contexts with doc_id per query)
- Qrels TSV (header: query-id, corpus-id, score)

Outputs:
- JSON with aggregate NDCG@10 and Recall@10 (and per-query if desired)
"""
import json
from pathlib import Path
from typing import Dict, List

import click

# Local imports via project root
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation_metrics import RetrievalQualityEvaluator


def load_qrels_tsv(path: Path) -> Dict[str, Dict[str, float]]:
    rel: Dict[str, Dict[str, float]] = {}
    with path.open('r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        # Expect columns like: query-id, corpus-id, score
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            qid, docid, score = parts[0], parts[1], parts[2]
            rel.setdefault(str(qid), {})[str(docid)] = float(score)
    return rel


@click.command()
@click.option('--results', required=True, help='Results JSON from run_reranking_experiment.py')
@click.option('--qrels', required=True, help='BEIR-style qrels TSV with header: query-id, corpus-id, score')
@click.option('--k', default=10, show_default=True, help='K for NDCG@K and Recall@K')
@click.option('--output', required=True, help='Output JSON path for evaluation metrics')
def main(results: str, qrels: str, k: int, output: str):
    res_path = Path(results)
    qrels_path = Path(qrels)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with res_path.open('r') as f:
        res_json = json.load(f)
    items = res_json['results'] if isinstance(res_json, dict) and 'results' in res_json else res_json

    # Build query_results mapping: {query_id: [docid, ...]}
    query_results: Dict[str, List[str]] = {}
    skipped = 0
    for r in items:
        qid = r.get('query_id')
        ctxs = r.get('contexts', [])
        if not qid:
            skipped += 1
            continue  # Require query IDs for BEIR alignment
        doc_ids = []
        for c in ctxs:
            # Prefer deriving BEIR corpus-id from metadata filename/source when available
            did = None
            if isinstance(c, dict):
                meta = c.get('metadata') or {}
                fname = meta.get('filename') or meta.get('source')
                if fname:
                    try:
                        from pathlib import Path as _P
                        did = _P(str(fname)).stem  # e.g., '12345.txt' -> '12345'
                    except Exception:
                        did = None
                if did is None:
                    # Fallbacks
                    did = meta.get('doc_id') or c.get('doc_id')
            if did is not None:
                doc_ids.append(str(did))
        query_results[str(qid)] = doc_ids

    ground = load_qrels_tsv(qrels_path)
    evaluator = RetrievalQualityEvaluator(ground)
    # Compute metrics
    ndcg_sum = 0.0
    recall_sum = 0.0
    count = 0
    per_query = {}
    for qid, docs in query_results.items():
        nd = evaluator.calculate_ndcg_at_k(qid, docs, k)
        rc = evaluator.calculate_recall_at_k(qid, docs, k)
        per_query[qid] = {'ndcg@k': nd, 'recall@k': rc}
        ndcg_sum += nd
        recall_sum += rc
        count += 1

    out = {
        'queries_evaluated': count,
        'skipped_missing_ids': skipped,
        'k': k,
        'ndcg@k_mean': (ndcg_sum / count) if count else 0.0,
        'recall@k_mean': (recall_sum / count) if count else 0.0,
        'per_query': per_query,
    }
    with out_path.open('w') as f:
        json.dump(out, f, indent=2)
    click.echo(f"Saved evaluation → {out_path}")


if __name__ == '__main__':
    main()
