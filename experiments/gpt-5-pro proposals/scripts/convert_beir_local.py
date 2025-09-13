#!/usr/bin/env python3
"""
Convert a locally unzipped BEIR-style dataset (corpus.jsonl, queries.jsonl, qrels/{train,dev,test}.tsv)
into the project layout expected by e1_eval.py:

out_dir/
  docs/<docid>.txt     # title + two newlines + text (or text only)
  queries.jsonl        # {"qid": str, "text": str}
  qrels.tsv            # qid\t0\tdocid\trelevance (uses test split by default)

Usage:
  python experiments/gpt-5-pro proposals/scripts/convert_beir_local.py \
    --in /path/to/unzipped/<name> \
    --out "experiments/gpt-5-pro proposals/datasets/<name>" \
    [--qrels-split test]

Notes:
  - corpus.jsonl lines typically have keys: _id, title, text
  - queries.jsonl lines typically have keys: _id, text
  - qrels split files have columns: qid, docid, score (TSV with optional header)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def convert_dataset(in_dir: Path, out_dir: Path, qrels_split: str = "test") -> None:
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    out_docs = out_dir / "docs"
    out_docs.mkdir(parents=True, exist_ok=True)

    # 1) Documents
    corpus_path = in_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus.jsonl at {corpus_path}")
    n_docs = 0
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("_id") or obj.get("doc_id") or obj.get("docid") or "").strip()
            if not doc_id:
                continue
            title = (obj.get("title") or "").strip()
            text = (obj.get("text") or "").strip()
            body = (f"{title}\n\n{text}" if title else text).strip()
            (out_docs / f"{doc_id}.txt").write_text(body, encoding="utf-8", errors="ignore")
            n_docs += 1

    # 2) Queries
    src_queries = in_dir / "queries.jsonl"
    if not src_queries.exists():
        raise FileNotFoundError(f"Missing queries.jsonl at {src_queries}")
    dst_queries = out_dir / "queries.jsonl"
    with src_queries.open("r", encoding="utf-8") as fin, dst_queries.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id") or obj.get("query_id") or obj.get("qid") or "").strip()
            text = (obj.get("text") or "").strip()
            if not qid:
                continue
            fout.write(json.dumps({"qid": qid, "text": text}, ensure_ascii=False) + "\n")

    # 3) Qrels
    split_path = in_dir / "qrels" / f"{qrels_split}.tsv"
    if not split_path.exists():
        # fallback: sometimes only qrels.tsv exists
        split_path = in_dir / "qrels.tsv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing qrels split file at {in_dir}/qrels/{qrels_split}.tsv")
    dst_qrels = out_dir / "qrels.tsv"
    with split_path.open("r", encoding="utf-8") as fin, dst_qrels.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            # Skip header-like lines
            if parts and any(h in parts[0].lower() for h in ["qid", "query", "query-id", "query_id"]):
                continue
            if len(parts) >= 4:
                qid, _zero_or_q0, docid, score = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) == 3:
                qid, docid, score = parts[0], parts[1], parts[2]
            else:
                continue
            # Validate score is numeric; skip if not
            try:
                float(score)
            except Exception:
                continue
            fout.write(f"{qid}\t0\t{docid}\t{score}\n")

    print(f"Converted {n_docs} docs into {out_docs}")
    print(f"Wrote queries to {dst_queries}")
    print(f"Wrote qrels to {dst_qrels}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert local BEIR dataset into project layout")
    ap.add_argument("--in", dest="in_dir", required=True, help="Path to unzipped dataset root (contains corpus.jsonl, queries.jsonl, qrels/")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output directory for project-formatted dataset")
    ap.add_argument("--qrels-split", default="test", choices=["train", "dev", "test"], help="Which qrels split to use")
    args = ap.parse_args()

    convert_dataset(Path(args.in_dir), Path(args.out_dir), qrels_split=args.qrels_split)


if __name__ == "__main__":
    main()
