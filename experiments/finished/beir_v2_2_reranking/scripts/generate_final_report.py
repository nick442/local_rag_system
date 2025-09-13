#!/usr/bin/env python3
"""
Generate a comprehensive report for BEIR v2.2 reranking experiments.

Consumes the JSON summary produced by summarize_results.py and writes a
FINAL_REPORT_v2_2_BEIR.md with methods, results, and analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import click


@dataclass
class Row:
    dataset: str
    run_name: str
    ndcg_at_10: float
    recall_at_10: float
    queries: int
    retrieval_method: Optional[str]
    alpha: Optional[float]
    candidate_multiplier: Optional[int]
    reranker_model: Optional[str]
    rerank_topk: Optional[int]


def load_rows(summary_json: Path) -> List[Row]:
    data = json.loads(summary_json.read_text())
    rows: List[Row] = []
    for d in data:
        rows.append(Row(
            dataset=d.get('dataset','unknown'),
            run_name=d.get('run_name',''),
            ndcg_at_10=float(d.get('ndcg_at_10') or 0.0),
            recall_at_10=float(d.get('recall_at_10') or 0.0),
            queries=int(d.get('queries') or 0),
            retrieval_method=d.get('retrieval_method'),
            alpha=d.get('alpha'),
            candidate_multiplier=d.get('candidate_multiplier'),
            reranker_model=d.get('reranker_model'),
            rerank_topk=d.get('rerank_topk'),
        ))
    return rows


def find_baseline(rows: List[Row], dataset: str) -> Optional[Row]:
    # Prefer exact baseline name, else heuristic
    for r in rows:
        if r.dataset == dataset and r.run_name.endswith('baseline'):
            return r
    for r in rows:
        if r.dataset == dataset and 'baseline' in r.run_name:
            return r
    # Fallback to lowest-complexity non-reranked row
    candidates = [r for r in rows if r.dataset == dataset and not r.reranker_model]
    return candidates[0] if candidates else None


def best_by_ndcg(rows: List[Row], dataset: str) -> Optional[Row]:
    subset = [r for r in rows if r.dataset == dataset]
    if not subset:
        return None
    return sorted(subset, key=lambda x: x.ndcg_at_10, reverse=True)[0]


def make_table(rows: List[Row], limit: int = 8) -> str:
    hdr = "| run | ndcg@10 | recall@10 | method | alpha | cm | reranker | topk |\n|---|---:|---:|---|---:|---:|---|---:|"
    def fmt(r: Row) -> str:
        return (
            f"| `{r.run_name}` | {r.ndcg_at_10:.4f} | {r.recall_at_10:.4f} | "
            f"{r.retrieval_method or '-'} | {r.alpha if r.alpha is not None else '-'} | "
            f"{r.candidate_multiplier if r.candidate_multiplier is not None else '-'} | "
            f"{r.reranker_model or '-'} | {r.rerank_topk if r.rerank_topk is not None else '-'} |"
        )
    body = "\n".join(fmt(r) for r in rows[:limit])
    return f"{hdr}\n{body}"


@click.command()
@click.option('--summary-json', default='experiments/reranking/results/summary.json', show_default=True,
              help='Path to summary JSON produced by summarize_results.py')
@click.option('--output', default='experiments/reranking/FINAL_REPORT_v2_2_BEIR.md', show_default=True,
              help='Output Markdown report path')
def main(summary_json: str, output: str):
    s = Path(summary_json)
    out = Path(output)
    rows = load_rows(s)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Sections for FiQA and SciFact
    fiqa_base = find_baseline(rows, 'fiqa')
    scifact_base = find_baseline(rows, 'scifact')
    fiqa_best = best_by_ndcg(rows, 'fiqa')
    scifact_best = best_by_ndcg(rows, 'scifact')

    def delta(a: Optional[Row], b: Optional[Row]) -> str:
        if not a or not b:
            return "n/a"
        dn = (b.ndcg_at_10 - a.ndcg_at_10)
        dr = (b.recall_at_10 - a.recall_at_10)
        return f"Δ ndcg={dn:+.4f}, Δ recall={dr:+.4f}"

    fiqa_rows = sorted([r for r in rows if r.dataset == 'fiqa'], key=lambda x: x.ndcg_at_10, reverse=True)
    scifact_rows = sorted([r for r in rows if r.dataset == 'scifact'], key=lambda x: x.ndcg_at_10, reverse=True)

    report = []
    report.append("# BEIR v2.2 Reranking Experiments — Final Report")
    report.append("")
    report.append(f"Generated: {ts}")
    report.append("")
    report.append("## Abstract")
    report.append("We evaluate two-stage retrieval with cross-encoder reranking on BEIR FiQA and SciFact corpora. "
                  "Using test-only queries aligned to BEIR qrels and k=10 evaluation, FiQA benefits from hybrid "
                  "first-stage retrieval combined with MiniLM-L12 reranking, improving NDCG and recall over vector-only. "
                  "SciFact shows strong gains in NDCG from the BAAI/BGE base reranker with a vector-only first stage.")
    report.append("")
    report.append("## Datasets and Setup")
    report.append("- Datasets: FiQA (fiqa_technical), SciFact (scifact_scientific)")
    report.append("- Database: data/rag_vectors.db; k=10 for final evaluation; rerank-topk ∈ {50,100}")
    report.append("- Queries: test-only BEIR queries with IDs (test_data/beir_*_queries_test_only.json)")
    report.append("- Metrics: NDCG@10, Recall@10 vs BEIR qrels (evaluate_reranking.py)")
    report.append("- First-stage: vector, hybrid(alpha, candidate-multiplier)")
    report.append("- Rerankers: cross-encoder/ms-marco-MiniLM-L-12-v2, BAAI/bge-reranker-base")
    report.append("")
    report.append("## Results")
    if fiqa_base and fiqa_best:
        report.append(f"- FiQA baseline: ndcg={fiqa_base.ndcg_at_10:.4f}, recall={fiqa_base.recall_at_10:.4f}")
        report.append(f"- FiQA best: `{fiqa_best.run_name}` ndcg={fiqa_best.ndcg_at_10:.4f}, recall={fiqa_best.recall_at_10:.4f} ({delta(fiqa_base, fiqa_best)})")
    if scifact_base and scifact_best:
        report.append(f"- SciFact baseline: ndcg={scifact_base.ndcg_at_10:.4f}, recall={scifact_base.recall_at_10:.4f}")
        report.append(f"- SciFact best: `{scifact_best.run_name}` ndcg={scifact_best.ndcg_at_10:.4f}, recall={scifact_best.recall_at_10:.4f} ({delta(scifact_base, scifact_best)})")
    report.append("")
    report.append("### FiQA — Top Runs (by NDCG@10)")
    report.append(make_table(fiqa_rows, limit=10))
    report.append("")
    report.append("### SciFact — Top Runs (by NDCG@10)")
    report.append(make_table(scifact_rows, limit=10))
    report.append("")
    report.append("## Discussion")
    report.append("- FiQA: Hybrid first-stage (alpha≈0.3–0.7, cand_mult=5) with MiniLM-L12 reranking yields the best NDCG and improves recall modestly; larger candidate sets (cm=10, topk=100) trade NDCG for recall.")
    report.append("- FiQA: BGE-base underperforms MiniLM-L12 on this domain/configuration; further tuning (instruction-tuned rerankers or larger BGE variants) may be required.")
    report.append("- SciFact: Vector-only first stage with BGE-base reranking substantially improves NDCG; recall does not match a saturated vector baseline when reranking top-k is small; increasing rerank-topk may trade latency for recall.")
    report.append("")
    report.append("## Limitations and Notes")
    report.append("- Evaluator maps corpus-ids via filename/source stem; ensure sources preserve BEIR doc IDs.")
    report.append("- FTS5 warnings during keyword search are expected and safe if results are produced.")
    report.append("- Latency profiling is out of scope here; expected overhead rises with rerank-topk and model size.")
    report.append("- Deduplicated metrics are provided for reference; main comparisons use non-dedup.")
    report.append("")
    report.append("## Reproducibility")
    report.append("- Env: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env`")
    report.append("- Status: `RAG_SWEEP_NO_LLM=1 python main.py --db-path data/rag_vectors.db status`")
    report.append("- Runner (example FiQA hybrid+L12): `RAG_SWEEP_NO_LLM=1 python experiments/reranking/run_reranking_experiment.py --queries test_data/beir_fiqa_queries_test_only.json --corpus fiqa_technical --k 10 --retrieval-method hybrid --alpha 0.5 --candidate-multiplier 5 --rerank-topk 50 --reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2 --output experiments/reranking/results/fiqa_hybrid_a0.5_cm5_miniLM12.topk50.test.json`")
    report.append("- Evaluate: `python experiments/reranking/evaluate_reranking.py --results <results.json> --qrels <qrels.tsv> --k 10 --output <metrics.json>`")
    report.append("- Summarize: `python experiments/reranking/summarize_results.py --results-dir experiments/reranking/results`")
    report.append("")
    report.append("## Artifacts")
    report.append("- Results and metrics JSONs under `experiments/reranking/results/`")
    report.append("- Summary: `experiments/reranking/results/summary.json`, `summary.md`")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report))
    click.echo(f"Wrote report → {out}")


if __name__ == '__main__':
    main()

