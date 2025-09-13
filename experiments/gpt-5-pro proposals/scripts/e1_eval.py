#!/usr/bin/env python3
"""
E1 Evaluation Runner (doc-aware ranking + latency/position diagnostics)

Runs retrieval evaluation for a given collection and writes metrics.
Also emits lightweight latency and position-sensitivity diagnostics to
aid E1 analysis without changing downstream aggregators.

Usage (examples):
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
  python scripts/e1_eval.py \
    --db-path data/rag_vectors__trec_covid_fixed_256_20.db \
    --embedding-path models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf \
    --collection trec_covid_fixed_256_20 \
    --dataset-dir "experiments/gpt-5-pro proposals/datasets/trec_covid" \
    --mode dense \
    --kdocs 10 --nchunks 100 --cap 3 \
    --outdir results/e1_trec_covid/full/trec_covid_fixed_256_20

Modes:
  dense  -> vector search
  bm25   -> keyword (FTS5 BM25)
  hybrid -> combined (z-normalized) vector+keyword
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
import statistics


def load_queries(path: Path) -> List[Tuple[str, str]]:
    import json
    queries = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            o = json.loads(line)
            queries.append((o['qid'], o['text']))
    return queries


def load_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    """Load qrels tolerant of different BEIR formats and headers.

    Accepts either 3-col (qid, docid, score) or 4-col (qid, 0, docid, score).
    Skips header rows and any lines that cannot be parsed as floats.
    """
    truth: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = s.split('\t')
            qid = docid = None
            rel_val: float | None = None
            try:
                if len(parts) >= 4:
                    qid, _zero, docid, rel = parts[0], parts[1], parts[2], parts[3]
                elif len(parts) == 3:
                    qid, docid, rel = parts[0], parts[1], parts[2]
                else:
                    continue
                # Skip typical headers (e.g., 'qid', 'query-id', etc.)
                if not qid or not docid or not rel:
                    continue
                # Header guard: qid/docid should not contain dashes/letters in canonical BEIR ids,
                # but be generous and rely on rel being numeric.
                rel_val = float(rel)
            except Exception:
                continue
            truth[qid][docid] = rel_val  # type: ignore[assignment]
    return truth


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def doc_aware_rank(
    rag_retriever,
    query: str,
    mode: str,
    collection_id: str,
    nchunks: int,
    cap: int,
    kdocs: int,
    return_diag: bool = False,
):
    """
    Doc-aware fold: cap chunks per doc, rank by best chunk score.
    Optionally returns diagnostics including best-chunk positions and retrieval latency.
    """
    method = 'vector' if mode == 'dense' else ('keyword' if mode == 'bm25' else 'hybrid')
    t0 = time.time()
    chunks = rag_retriever.retrieve(query, k=nchunks, method=method, collection_id=collection_id)
    latency_ms = (time.time() - t0) * 1000.0

    pool: Dict[str, List] = defaultdict(list)
    for ch in chunks:
        # Prefer doc_id from result; fall back to filename stem if missing
        doc_key = ch.doc_id
        if not doc_key:
            from pathlib import Path as P
            meta = ch.metadata or {}
            fname = meta.get('filename') or meta.get('source') or ''
            doc_key = P(fname).stem if fname else ''
        if not doc_key:
            continue
        if len(pool[doc_key]) < cap:
            pool[doc_key].append(ch)

    scored: List[Tuple[str, float]] = []
    diag_list = []
    # For position analysis we will compute the best chunk index and doc length when asked
    for dk, items in pool.items():
        best = max(items, key=lambda r: r.score)
        scored.append((dk, best.score))
        if return_diag:
            try:
                # Retrieve doc length (number of chunks) to normalize position
                doc_chunks = rag_retriever.vector_db.get_document_chunks(best.doc_id)
                doc_len = len(doc_chunks) if doc_chunks else 0
            except Exception:
                doc_len = 0
            rel_pos = (best.chunk_index / max(1, doc_len - 1)) if doc_len > 1 else 0.0
            diag_list.append({
                'doc_id': dk,
                'best_chunk_index': best.chunk_index,
                'doc_len': doc_len,
                'relative_pos': rel_pos,
                'best_score': best.score,
            })

    ranked_pairs = sorted(scored, key=lambda x: x[1], reverse=True)[:kdocs]
    ranked = [d for d, _ in ranked_pairs]
    if return_diag:
        # Keep diagnostics aligned to ranked doc order
        diag_by_doc = {d['doc_id']: d for d in diag_list}
        ranked_diag = [diag_by_doc.get(doc_id) for doc_id in ranked if doc_id in diag_by_doc]
        return ranked, ranked_diag, latency_ms
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db-path', required=True)
    ap.add_argument('--embedding-path', required=True)
    ap.add_argument('--collection', required=True)
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--queries', default=None, help='Path to queries.jsonl (defaults to <dataset-dir>/queries.jsonl)')
    ap.add_argument('--qrels', default=None, help='Path to qrels.tsv (defaults to <dataset-dir>/qrels.tsv)')
    ap.add_argument('--mode', choices=['dense', 'bm25', 'hybrid'], default='dense')
    ap.add_argument('--kdocs', type=int, default=10)
    ap.add_argument('--nchunks', type=int, default=100)
    ap.add_argument('--cap', type=int, default=3)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--emit_diagnostics', action='store_true', help='Emit latency and position diagnostics alongside metrics')
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    queries_path = Path(args.queries) if args.queries else dataset_dir / 'queries.jsonl'
    qrels_path = Path(args.qrels) if args.qrels else dataset_dir / 'qrels.tsv'
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(queries_path)
    truth = load_qrels(qrels_path)

    # Prepare RAG retriever
    import sys
    sys.path.insert(0, str(Path('.').resolve()))
    from src.rag_pipeline import RAGPipeline
    from src.evaluation_metrics import RetrievalQualityEvaluator

    rag = RAGPipeline(db_path=args.db_path, embedding_model_path=args.embedding_path, llm_model_path='models/gemma-3-4b-it-q4_0.gguf')

    results_docs: Dict[str, List[str]] = {}
    per_query_records: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    for qid, text in queries:
        if args.emit_diagnostics:
            ranked, ranked_diag, latency_ms = doc_aware_rank(
                rag.retriever, text, args.mode, args.collection, args.nchunks, args.cap, args.kdocs, return_diag=True
            )
            latencies_ms.append(latency_ms)
            per_query_records.append({
                'qid': qid,
                'ranked_docs': ranked,
                'latency_ms': latency_ms,
                'diagnostics': ranked_diag,
            })
        else:
            ranked = doc_aware_rank(rag.retriever, text, args.mode, args.collection, args.nchunks, args.cap, args.kdocs)
        results_docs[qid] = ranked

    rqe = RetrievalQualityEvaluator(ground_truth_relevance=truth)
    metrics = rqe.evaluate_query_set(results_docs, k_values=[1, 5, 10, 20])

    # Write outputs
    suffix = '' if args.mode == 'dense' else f'_{args.mode}'
    (outdir / f'metrics{suffix}.json').write_text(json.dumps(metrics, indent=2))
    print('Wrote', outdir / f'metrics{suffix}.json')

    # Optional diagnostics: latency + position analysis
    if args.emit_diagnostics:
        # Latency summary
        lat_summary = {
            'per_query_ms': latencies_ms,
            'mean_ms': statistics.mean(latencies_ms) if latencies_ms else 0.0,
            'p50_ms': _percentile(latencies_ms, 50.0) if latencies_ms else 0.0,
            'p95_ms': _percentile(latencies_ms, 95.0) if latencies_ms else 0.0,
        }
        (outdir / f'latency{suffix}.json').write_text(json.dumps(lat_summary, indent=2))

        # Position sensitivity (top-1 doc per query if available)
        top1_rel_positions: List[float] = []
        for rec in per_query_records:
            diags = rec.get('diagnostics') or []
            if not diags:
                continue
            top1 = diags[0]
            if top1 and isinstance(top1.get('relative_pos'), (int, float)):
                top1_rel_positions.append(float(top1['relative_pos']))
        # Simple histogram bins for begin/middle/end
        bins = {
            'begin_0_0_2': sum(1 for v in top1_rel_positions if 0.0 <= v < 0.2),
            'early_0_2_0_4': sum(1 for v in top1_rel_positions if 0.2 <= v < 0.4),
            'middle_0_4_0_6': sum(1 for v in top1_rel_positions if 0.4 <= v < 0.6),
            'late_0_6_0_8': sum(1 for v in top1_rel_positions if 0.6 <= v < 0.8),
            'end_0_8_1_0': sum(1 for v in top1_rel_positions if 0.8 <= v <= 1.0),
        }
        pos_summary = {
            'top1_relative_positions': top1_rel_positions,
            'mean_top1_relative_position': statistics.mean(top1_rel_positions) if top1_rel_positions else 0.0,
            'histogram_top1': bins,
            'n': len(top1_rel_positions),
        }
        (outdir / f'position{suffix}.json').write_text(json.dumps(pos_summary, indent=2))

        # Per-query records for downstream analysis
        with (outdir / f'per_query{suffix}.jsonl').open('w', encoding='utf-8') as f:
            for rec in per_query_records:
                f.write(json.dumps(rec) + "\n")


if __name__ == '__main__':
    main()
