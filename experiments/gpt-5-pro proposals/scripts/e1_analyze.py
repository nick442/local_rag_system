#!/usr/bin/env python3
"""
E1 Analysis Script: aggregates metrics across variants/modes and produces a summary CSV + JSON.

Inputs (hard-coded defaults target our E1 run):
  - results/e1_trec_covid/full/trec_covid_fixed_256_20/metrics*.json
  - results/e1_trec_covid/full/trec_covid_fixed_512_50/metrics*.json

Outputs:
  - results/e1_trec_covid/full/summary/summary.csv
  - results/e1_trec_covid/full/summary/summary.json
  - results/e1_trec_covid/full/summary/diff_dense_256_vs_512.json (effect size, mean deltas when per-query present)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any


def load_metrics(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def pick(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract common fields for the CSV row."""
    p1 = summary.get('precision_at_k', {}).get('1', {}).get('mean')
    p10 = summary.get('precision_at_k', {}).get('10', {}).get('mean')
    ndcg10 = summary.get('ndcg_at_k', {}).get('10', {}).get('mean')
    mrr = summary.get('mrr')
    return {
        'P@1': p1,
        'P@10': p10,
        'NDCG@10': ndcg10,
        'MRR': mrr,
    }


def paired_diff_ndcg10(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Compute paired comparison for NDCG@10 if per_query_results exist in both."""
    try:
        a_map = {qid: vals.get('ndcg_at_10') for qid, vals in a.get('per_query_results', {}).items()}
        b_map = {qid: vals.get('ndcg_at_10') for qid, vals in b.get('per_query_results', {}).items()}
        common = [qid for qid in a_map.keys() if qid in b_map]
        a_vals = [a_map[qid] for qid in common]
        b_vals = [b_map[qid] for qid in common]
        # Use the repo's StatisticalAnalyzer for effect size/t-stat if available
        from src.evaluation_metrics import StatisticalAnalyzer
        stats = StatisticalAnalyzer.paired_comparison(a_vals, b_vals)
        stats['n_pairs'] = len(common)
        stats['mean_a'] = sum(a_vals)/len(a_vals) if a_vals else 0.0
        stats['mean_b'] = sum(b_vals)/len(b_vals) if b_vals else 0.0
        stats['metric'] = 'ndcg@10'
        return stats
    except Exception:
        return {'error': 'per_query_results missing or incompatible', 'metric': 'ndcg@10'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results/e1_trec_covid/full/summary')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = Path('results/e1_trec_covid/full')
    variants = {
        'trec_covid_fixed_256_20': base / 'trec_covid_fixed_256_20',
        'trec_covid_fixed_512_50': base / 'trec_covid_fixed_512_50',
    }
    modes = ['dense', 'bm25', 'hybrid']

    rows = []
    summary_json = {}

    for var, vdir in variants.items():
        summary_json[var] = {}
        for mode in modes:
            fn = 'metrics.json' if mode == 'dense' else f'metrics_{mode}.json'
            path = vdir / fn
            m = load_metrics(path)
            if not m:
                continue
            key = f'{var}/{mode}'
            mm = pick(m)
            mm['variant'] = var
            mm['mode'] = mode
            rows.append(mm)
            summary_json[key] = mm

    # Write CSV
    csv_path = outdir / 'summary.csv'
    if rows:
        with csv_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['variant', 'mode', 'P@1', 'P@10', 'NDCG@10', 'MRR'])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Write JSON
    (outdir / 'summary.json').write_text(json.dumps(summary_json, indent=2))

    # Paired diff: dense 256/20 vs 512/50 on NDCG@10
    a = load_metrics(variants['trec_covid_fixed_256_20'] / 'metrics.json')
    b = load_metrics(variants['trec_covid_fixed_512_50'] / 'metrics.json')
    if a and b:
        diff = paired_diff_ndcg10(a, b)
        (outdir / 'diff_dense_256_vs_512.json').write_text(json.dumps(diff, indent=2))

    print('Wrote', csv_path, 'and JSON summaries to', outdir)


if __name__ == '__main__':
    main()

