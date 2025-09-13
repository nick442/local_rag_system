#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

def load_queries(p: Path):
    qs=[]
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            o=json.loads(line); qs.append((o['qid'], o['text']))
    return qs

def load_qrels(p: Path):
    truth=defaultdict(dict)
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            qid,_zero,docid,rel=line.strip().split('\t')
            truth[qid][docid]=float(rel)
    return truth

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db-path', required=True)
    ap.add_argument('--embedding-path', required=True)
    ap.add_argument('--collection', required=True)
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--nchunks', type=int, default=2000)
    ap.add_argument('--mdocs', type=int, default=50)
    ap.add_argument('--kdocs', type=int, default=10)
    ap.add_argument('--cap', type=int, default=3)
    ap.add_argument('--outdir', required=True)
    args=ap.parse_args()

    ds=Path(args.dataset_dir)
    queries=load_queries(ds/'queries.jsonl')
    truth=load_qrels(ds/'qrels.tsv')
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    import sys; sys.path.insert(0, str(Path('.').resolve()))
    from src.rag_pipeline import RAGPipeline
    from src.evaluation_metrics import RetrievalQualityEvaluator
    from pathlib import Path as P

    rag=RAGPipeline(db_path=args.db_path, embedding_model_path=args.embedding_path, llm_model_path='models/gemma-3-4b-it-q4_0.gguf')

    results_docs={}
    for qid,text in queries:
        chunks=rag.retriever.retrieve(text, k=args.nchunks, method='vector', collection_id=args.collection)
        doc_best=defaultdict(float); doc_chunks=defaultdict(list)
        for ch in chunks:
            meta=ch.metadata or {}; fname=meta.get('filename') or meta.get('source') or ''
            dk=P(fname).stem if fname else ''
            if not dk: continue
            if ch.score>doc_best[dk]: doc_best[dk]=ch.score
            doc_chunks[dk].append(ch)
        top_docs=[d for d,_ in sorted(doc_best.items(), key=lambda x:x[1], reverse=True)[:args.mdocs]]
        pool=defaultdict(list)
        for dk in top_docs:
            for ch in doc_chunks.get(dk, []):
                if len(pool[dk])<args.cap: pool[dk].append(ch)
        scored=[]
        for dk, items in pool.items():
            if not items: continue
            best=max(items, key=lambda r:r.score)
            scored.append((dk, best.score))
        ranked=[d for d,_ in sorted(scored, key=lambda x:x[1], reverse=True)[:args.kdocs]]
        results_docs[qid]=ranked

    rqe=RetrievalQualityEvaluator(ground_truth_relevance=truth)
    metrics=rqe.evaluate_query_set(results_docs, k_values=[1,5,10,20])
    (outdir/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Wrote', outdir/'metrics.json')

if __name__=='__main__':
    main()

