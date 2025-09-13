#!/usr/bin/env python3
"""
Summarize reranking experiment metrics into JSON and Markdown tables.

Scans a results directory for *.metrics.json files (produced by
experiments/reranking/evaluate_reranking.py) and joins each with its
corresponding *.test.json results file to extract retrieval settings
like method, alpha, candidate-multiplier, reranker-model, and rerank-topk.

Outputs:
- JSON summary with one row per metrics file
- Optional Markdown file with tables grouped by dataset
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click


@dataclass
class RunSummary:
    dataset: str
    run_name: str
    metrics_path: str
    results_path: Optional[str]
    ndcg_at_10: float
    recall_at_10: float
    queries: int
    retrieval_method: Optional[str] = None
    alpha: Optional[float] = None
    candidate_multiplier: Optional[int] = None
    reranker_model: Optional[str] = None
    rerank_topk: Optional[int] = None
    deduplicated: bool = False


def infer_dataset_from_name(name: str) -> str:
    lname = name.lower()
    if lname.startswith("fiqa_"):
        return "fiqa"
    if lname.startswith("scifact_"):
        return "scifact"
    if "fiqa" in lname:
        return "fiqa"
    if "scifact" in lname:
        return "scifact"
    return "unknown"


def find_results_for_metrics(metrics_path: Path) -> Optional[Path]:
    # Try replacing .metrics.json with .test.json
    base = metrics_path.name
    if base.endswith(".metrics.json"):
        candidate = metrics_path.with_name(base.replace(".metrics.json", ".test.json"))
        if candidate.exists():
            return candidate
        # Some files include .dedup.metrics.json
        candidate = metrics_path.with_name(base.replace(".dedup.metrics.json", ".test.json"))
        if candidate.exists():
            return candidate
    return None


def extract_retrieval_settings(results_path: Path) -> Tuple[Optional[str], Optional[float], Optional[int], Optional[str], Optional[int]]:
    try:
        data = json.loads(results_path.read_text())
        items = data['results'] if isinstance(data, dict) and 'results' in data else data
        if not items:
            return None, None, None, None, None
        first = items[0]
        r = first.get('retrieval', {}) if isinstance(first, dict) else {}
        method = r.get('method')
        alpha = r.get('alpha')
        cand = r.get('candidate_multiplier')
        reranker = r.get('reranker_model')
        topk = r.get('rerank_topk')
        return method, alpha, cand, reranker, topk
    except Exception:
        return None, None, None, None, None


def collect_summaries(results_dir: Path) -> List[RunSummary]:
    rows: List[RunSummary] = []
    for metrics_path in sorted(results_dir.glob("*.metrics.json")):
        try:
            m = json.loads(metrics_path.read_text())
        except Exception:
            continue

        name = metrics_path.stem  # e.g., fiqa_hybrid_a0.3_cm5_miniLM12.topk50.metrics
        dedup = ".dedup." in metrics_path.name
        run_name = name.replace(".metrics", "")
        dataset = infer_dataset_from_name(metrics_path.name)
        ndcg = float(m.get("ndcg@k_mean") or 0.0)
        recall = float(m.get("recall@k_mean") or 0.0)
        queries = int(m.get("queries_evaluated") or 0)

        results_path = find_results_for_metrics(metrics_path)
        method = alpha = cand = reranker = topk = None
        if results_path and results_path.exists():
            method, alpha, cand, reranker, topk = extract_retrieval_settings(results_path)

        rows.append(RunSummary(
            dataset=dataset,
            run_name=run_name,
            metrics_path=str(metrics_path),
            results_path=str(results_path) if results_path else None,
            ndcg_at_10=ndcg,
            recall_at_10=recall,
            queries=queries,
            retrieval_method=method,
            alpha=alpha,
            candidate_multiplier=cand,
            reranker_model=reranker,
            rerank_topk=topk,
            deduplicated=dedup,
        ))
    return rows


def make_markdown(rows: List[RunSummary]) -> str:
    def fmt_row(r: RunSummary) -> str:
        rr = r.reranker_model or "-"
        return (
            f"| `{r.run_name}` | {r.ndcg_at_10:.4f} | {r.recall_at_10:.4f} | {r.queries} | "
            f"{r.retrieval_method or '-'} | {r.alpha if r.alpha is not None else '-'} | {r.candidate_multiplier if r.candidate_multiplier is not None else '-'} | {rr} | {r.rerank_topk if r.rerank_topk is not None else '-'} |"
        )

    parts: List[str] = []
    for dataset in sorted({r.dataset for r in rows}):
        subset = [r for r in rows if r.dataset == dataset]
        if not subset:
            continue
        subset.sort(key=lambda x: x.ndcg_at_10, reverse=True)
        parts.append(f"## {dataset.upper()} Results\n")
        parts.append("| run | ndcg@10 | recall@10 | n | method | alpha | cand_mult | reranker | rerank_topk |")
        parts.append("|---|---:|---:|---:|---|---:|---:|---|---:|")
        parts.extend(fmt_row(r) for r in subset)
        parts.append("")
    return "\n".join(parts)


@click.command()
@click.option('--results-dir', default='experiments/reranking/results', show_default=True,
              help='Directory with *.metrics.json files')
@click.option('--output', default='experiments/reranking/results/summary.json', show_default=True,
              help='Output JSON summary path')
@click.option('--markdown', default='experiments/reranking/results/summary.md', show_default=True,
              help='Optional Markdown table output path')
def main(results_dir: str, output: str, markdown: str):
    rdir = Path(results_dir)
    rows = collect_summaries(rdir)
    rows_dicts = [asdict(r) for r in rows]
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows_dicts, indent=2))

    if markdown:
        md = make_markdown(rows)
        Path(markdown).write_text(md)

    click.echo(f"Wrote summary JSON → {out_path}")
    if markdown:
        click.echo(f"Wrote summary Markdown → {markdown}")


if __name__ == '__main__':
    main()

