#!/usr/bin/env python3
"""
Minimal runner for v2.2 reranking experiments.

Retrieval-only loop that compares baseline vs reranked contexts. Reranking is
enabled by setting the env var RAG_RERANKER_MODEL or passing --reranker-model.
"""
import json
import os
import sys
from pathlib import Path
from typing import List

import click

# Ensure project root is on sys.path when run as a file
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_manager import ConfigManager
from src.rag_pipeline import RAGPipeline


@click.command()
@click.option('--queries', required=True, help='JSON file with evaluation queries')
@click.option('--corpus', required=True, help='Target corpus/collection (e.g., fiqa_technical)')
@click.option('--k', default=5, show_default=True, help='Top-k contexts to return (final)')
@click.option('--db-path', default='data/rag_vectors.db', show_default=True)
@click.option('--reranker-model', default=None, help='CrossEncoder model name (enables reranking)')
@click.option('--rerank-topk', type=int, default=None, help='Optional top-k truncation after reranking')
@click.option('--retrieval-method', type=click.Choice(['vector', 'keyword', 'hybrid']), default='vector', show_default=True,
              help='First-stage retrieval method')
@click.option('--alpha', type=float, default=None, help='Hybrid fusion alpha (if method=hybrid)')
@click.option('--candidate-multiplier', type=int, default=None, help='Hybrid candidate multiplier (if method=hybrid)')
@click.option('--output', required=True, help='Output JSON file for results')
def main(queries: str, corpus: str, k: int, db_path: str, reranker_model: str, rerank_topk: int,
         retrieval_method: str, alpha: float, candidate_multiplier: int, output: str):
    cfg = ConfigManager()

    # Prefer explicit model; fallback to env var
    if reranker_model:
        os.environ['RAG_RERANKER_MODEL'] = reranker_model

    # Build retrieval-only pipeline
    rag = RAGPipeline(
        db_path=db_path,
        embedding_model_path=cfg.get_param('embedding_model_path', 'sentence-transformers/all-MiniLM-L6-v2'),
        llm_model_path=cfg.get_param('llm_model_path', 'models/gemma-3-4b-it-q4_0.gguf'),
        disable_llm=True,
        reranker_model=reranker_model,
        rerank_top_k=rerank_topk,
    )
    rag.set_corpus(corpus)

    # Load queries
    with open(queries, 'r') as f:
        q_list = json.load(f)
    if isinstance(q_list, dict) and 'queries' in q_list:
        q_list = q_list['queries']
    if not isinstance(q_list, list):
        raise click.BadParameter('Queries file must be a list or {"queries": [...]}')

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine first-stage retrieval size: ensure reranker sees enough candidates
    k_retrieve = max(k, rerank_topk or k)

    records: List[dict] = []
    for q in q_list:
        text = q['query'] if isinstance(q, dict) and 'query' in q else (q.get('text') if isinstance(q, dict) and 'text' in q else str(q))
        qid = None
        if isinstance(q, dict):
            qid = str(q.get('id') or q.get('query_id') or '').strip() or None
        res = rag.query(
            text,
            k=k_retrieve,
            retrieval_method=retrieval_method,
            similarity_threshold=alpha,
            candidate_multiplier=candidate_multiplier,
        )
        # Only keep final top-k contexts for evaluation
        ctxs = res.get('contexts', [])
        if isinstance(ctxs, list) and len(ctxs) > k:
            ctxs = ctxs[:k]
        records.append({
            'query': text,
            'query_id': qid,
            'contexts': ctxs,
            'metadata': res.get('metadata', {}),
            'retrieval': {
                'method': retrieval_method,
                'k_retrieve': k_retrieve,
                'alpha': alpha,
                'candidate_multiplier': candidate_multiplier,
                'reranker_model': reranker_model,
                'rerank_topk': rerank_topk,
            }
        })

    with out_path.open('w') as f:
        json.dump({'results': records}, f)

    click.echo(f"Saved {len(records)} results → {out_path}")


if __name__ == '__main__':
    main()
