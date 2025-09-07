#!/usr/bin/env python3
"""
Create an executive summary (Markdown) and a combined PDF of key charts.

Outputs:
- experiments/hybrid/EXEC_SUMMARY.md
- experiments/hybrid/figures/EXEC_SUMMARY.pdf (merged charts)
"""

from pathlib import Path
import json

def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def write_summary_md(root: Path):
    out = root / 'experiments' / 'hybrid' / 'EXEC_SUMMARY.md'
    fiqa = load_json(root / 'experiments' / 'hybrid' / 'analysis' / 'fiqa_alpha_analysis.json')
    sci = load_json(root / 'experiments' / 'hybrid' / 'analysis' / 'scifact_alpha_analysis.json')
    lines = []
    lines.append('# Executive Summary: Hybrid Retrieval Optimization (Full Test Sets)')
    for name, data in [('FiQA', fiqa), ('SciFact', sci)]:
        if not data:
            continue
        opt = data.get('optimal_alpha')
        stats = data.get('alpha_statistics', {})
        sopt = stats.get(str(opt)) or stats.get(opt, {})
        ndcg = sopt.get('avg_ndcg@10')
        recall = sopt.get('avg_recall@10')
        lines.append(f"\n## {name}")
        lines.append(f"- Optimal α: {opt}")
        if ndcg is not None:
            lines.append(f"- Avg NDCG@10 at α: {ndcg:.3f}")
        if recall is not None:
            lines.append(f"- Avg Recall@10 at α: {recall:.3f}")
    out.write_text('\n'.join(lines))
    print(f"Wrote {out}")

def merge_pdfs(root: Path):
    try:
        from PyPDF2 import PdfMerger
    except Exception:
        print('PyPDF2 not available; skipping PDF merge')
        return
    figs = root / 'experiments' / 'hybrid' / 'figures'
    inputs = [
        figs / 'summary_dashboard.pdf',
        figs / 'alpha_optimization_curves.pdf',
        figs / 'performance_heatmap.pdf',
        figs / 'hypothesis_validation.pdf',
    ]
    merger = PdfMerger()
    added = 0
    for p in inputs:
        if p.exists():
            merger.append(str(p))
            added += 1
    if added:
        outp = figs / 'EXEC_SUMMARY.pdf'
        merger.write(str(outp))
        merger.close()
        print(f"Wrote {outp}")
    else:
        print('No input PDFs found to merge')

if __name__ == '__main__':
    here = Path(__file__).resolve()
    # Ascend until we find main.py (project root)
    root = here
    for _ in range(8):
        if (root / 'main.py').exists():
            break
        root = root.parent
    write_summary_md(root)
    merge_pdfs(root)
