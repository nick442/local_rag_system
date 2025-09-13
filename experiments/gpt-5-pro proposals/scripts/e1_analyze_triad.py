#!/usr/bin/env python3
"""
E1 Triad Analysis: aggregates metrics across TREC-COVID, SciFact, FiQA variants/modes.

Inputs (by default):
  - results/e1_<dataset>/full/<collection>/metrics*.json
    where <dataset> in {trec_covid, scifact, fiqa}

Outputs:
  - results/e1_triad/summary.csv
  - results/e1_triad/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def load_metrics(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def pick(summary: Dict[str, Any]) -> Dict[str, Any]:
    p1 = summary.get('precision_at_k', {}).get('1', {}).get('mean')
    p10 = summary.get('precision_at_k', {}).get('10', {}).get('mean')
    ndcg10 = summary.get('ndcg_at_k', {}).get('10', {}).get('mean')
    mrr = summary.get('mrr')
    return {'P@1': p1, 'P@10': p10, 'NDCG@10': ndcg10, 'MRR': mrr}


def scan_dataset(dataset: str, base: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ds_root = base / f'e1_{dataset}' / 'full'
    if not ds_root.exists():
        return rows
    for coll_dir in ds_root.iterdir():
        if not coll_dir.is_dir():
            continue
        variant = coll_dir.name
        for mode in ['dense', 'bm25', 'hybrid']:
            # Prefer aliased metrics if present (doc IDs mapped to corpus IDs)
            candidates: List[str]
            if mode == 'dense':
                candidates = ['metrics_dense_aliased.json', 'metrics_aliased.json', 'metrics.json']
            else:
                candidates = [f'metrics_{mode}_aliased.json', f'metrics_{mode}.json']

            m = {}
            for name in candidates:
                path = coll_dir / name
                m = load_metrics(path)
                if m:
                    break
            if not m:
                continue
            row = pick(m)
            row.update({'dataset': dataset, 'variant': variant, 'mode': mode})
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='results', help='Results base directory')
    ap.add_argument('--outdir', default='results/e1_triad', help='Where to write summaries')
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = ['trec_covid', 'scifact', 'fiqa']
    all_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        all_rows.extend(scan_dataset(ds, base))

    # CSV
    csv_path = outdir / 'summary.csv'
    if all_rows:
        with csv_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['dataset', 'variant', 'mode', 'P@1', 'P@10', 'NDCG@10', 'MRR'])
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    # JSON
    summary_map: Dict[str, Any] = {}
    for r in all_rows:
        key = f"{r['dataset']}/{r['variant']}/{r['mode']}"
        summary_map[key] = r
    (outdir / 'summary.json').write_text(json.dumps(summary_map, indent=2))

    print('Wrote triad summary to', outdir)


if __name__ == '__main__':
    main()
