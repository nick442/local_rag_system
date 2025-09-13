#!/usr/bin/env python3
"""
E3 TREC-COVID Summary: aggregate RRF and Z-Score fusion metrics across variants.

Scans these default locations (smoke/test runs):
  - results/e1_trec_covid/test/trec_covid_fixed_256_20/
  - results/e1_trec_covid/test/trec_covid_fixed_512_50/
  - results/e1_trec_covid/test/trec_covid_semantic_256_0/

Writes:
  - results/e3_trec_covid/summary/{summary.csv, summary.json}
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='results/e1_trec_covid/test')
    ap.add_argument('--outdir', default='results/e3_trec_covid/summary')
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    variants = [
        'trec_covid_fixed_256_20',
        'trec_covid_fixed_512_50',
        'trec_covid_semantic_256_0',
    ]

    rows: List[Dict[str, Any]] = []
    summary_map: Dict[str, Any] = {}

    for var in variants:
        vdir = base / var
        for fusion in ['rrf', 'zscore']:
            mpath = vdir / f'metrics_{fusion}.json'
            m = load_metrics(mpath)
            if not m:
                continue
            rec = pick(m)
            rec.update({'variant': var, 'fusion': fusion})
            rows.append(rec)
            summary_map[f'{var}/{fusion}'] = rec

    # CSV
    csv_path = outdir / 'summary.csv'
    if rows:
        with csv_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['variant', 'fusion', 'P@1', 'P@10', 'NDCG@10', 'MRR'])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # JSON
    (outdir / 'summary.json').write_text(json.dumps(summary_map, indent=2))
    print('Wrote E3 TREC-COVID summary to', outdir)


if __name__ == '__main__':
    main()

