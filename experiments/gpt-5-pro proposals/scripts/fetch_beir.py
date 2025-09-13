#!/usr/bin/env python3
"""
Fetch BEIR datasets via ir_datasets and export to local layout:

datasets/<name>/
  docs/<docid>.txt      # title + body
  queries.jsonl         # {"qid": str, "text": str}
  qrels.tsv             # qid \t 0 \t docid \t relevance

Usage:
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && \
  python scripts/fetch_beir.py --dataset trec_covid

Notes:
- Uses ir_datasets. Ensure it’s installed in the active environment.
- Validates that all qrels docids exist in docs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set


def _resolve_beir_id(dataset: str) -> str:
    ds = dataset.lower().replace("-", "_")
    mapping = {
        "trec_covid": "beir/trec-covid",
        "scifact": "beir/scifact",
        "fiqa": "beir/fiqa-2018",
        "fiqa_2018": "beir/fiqa-2018",
    }
    if ds in mapping:
        return mapping[ds]
    # Fallback: assume caller passed a valid ir_datasets id
    return dataset


def export_beir(dataset: str, out_root: Path) -> None:
    try:
        import ir_datasets as irds
    except Exception as e:
        raise RuntimeError(
            "ir_datasets is not installed. Install it in your env: pip install ir_datasets"
        ) from e

    beir_id = _resolve_beir_id(dataset)
    ds = irds.load(beir_id)

    out_root.mkdir(parents=True, exist_ok=True)
    docs_dir = out_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    queries_path = out_root / "queries.jsonl"
    qrels_path = out_root / "qrels.tsv"

    # Export documents (title + text)
    print(f"[fetch_beir] Exporting documents to {docs_dir} …")
    count_docs = 0
    for doc in ds.docs_iter():
        doc_id = str(getattr(doc, "doc_id", getattr(doc, "docid", "")))
        if not doc_id:
            continue
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or ""
        body = (title.strip() + "\n\n" + text.strip()).strip() if title else text.strip()
        (docs_dir / f"{doc_id}.txt").write_text(body, encoding="utf-8", errors="ignore")
        count_docs += 1
        if count_docs % 10000 == 0:
            print(f"  … {count_docs} docs written")
    print(f"[fetch_beir] Documents written: {count_docs}")

    # Export queries
    print(f"[fetch_beir] Writing queries to {queries_path} …")
    with queries_path.open("w", encoding="utf-8") as f:
        count_q = 0
        for q in ds.queries_iter():
            qid = str(getattr(q, "query_id", getattr(q, "qid", "")))
            text = getattr(q, "text", "")
            if not qid:
                continue
            f.write(json.dumps({"qid": qid, "text": text}, ensure_ascii=False) + "\n")
            count_q += 1
    print(f"[fetch_beir] Queries written: {count_q}")

    # Export qrels
    print(f"[fetch_beir] Writing qrels to {qrels_path} …")
    with qrels_path.open("w", encoding="utf-8") as f:
        count_r = 0
        for rel in ds.qrels_iter():
            qid = str(getattr(rel, "query_id", getattr(rel, "qid", "")))
            doc_id = str(getattr(rel, "doc_id", getattr(rel, "docid", "")))
            score = getattr(rel, "relevance", getattr(rel, "label", 0))
            f.write(f"{qid}\t0\t{doc_id}\t{score}\n")
            count_r += 1
    print(f"[fetch_beir] Qrels written: {count_r}")

    # Validation: qrels docids are present in docs/
    print("[fetch_beir] Validating qrels doc IDs are present in docs/ …")
    doc_filenames: Set[str] = {p.stem for p in docs_dir.glob("*.txt")}
    missing = 0
    for rel in ds.qrels_iter():
        doc_id = str(getattr(rel, "doc_id", getattr(rel, "docid", "")))
        if doc_id and doc_id not in doc_filenames:
            missing += 1
            if missing <= 5:
                print(f"  MISSING docid in docs/: {doc_id}")
    if missing:
        raise RuntimeError(f"Validation failed: {missing} qrels doc IDs not found in docs/.")
    print("[fetch_beir] Validation passed: all qrels doc IDs present in docs/.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export BEIR datasets to local layout")
    ap.add_argument("--dataset", default="trec_covid", help="Dataset name (trec_covid|scifact|fiqa) or ir_datasets id")
    ap.add_argument("--out", default=None, help="Output directory (default: datasets/<name>)")
    args = ap.parse_args()

    name = args.dataset.lower().replace("-", "_")
    out = Path(args.out) if args.out else Path("datasets") / name
    export_beir(args.dataset, out)


if __name__ == "__main__":
    main()

