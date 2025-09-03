# Experiment 1 v2 — Staged Prep from realistic_full_production

This folder tracks staged preparation of per‑configuration collections cloned from the larger base corpus `realistic_full_production`.

## Stage Plan
1. Clone a small batch of target collections (no embeddings) to validate lifecycle and counts:
   - exp_cs256_co51 (~20%), exp_cs256_co64 (~25%)
   - exp_cs512_co102 (~20%), exp_cs512_co128 (~25%)
2. Re-embed one target (e.g., exp_cs256_co64) to validate vector search and retrieval.
3. Continue cloning remaining configs in batches; re-embed in subsequent stages.

## Notes
- Embedding path used later: models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<snapshot>
- Base collection: realistic_full_production

## Stage 1 — Clone (no embeddings)

Cloned the following from `realistic_full_production`:

- exp_full_cs256_co51: docs=10888, chunks=56014
- exp_full_cs256_co64: docs=10888, chunks=58256
- exp_full_cs512_co128: docs=10888, chunks=31610

Next stage: re-embed these collections to enable vector search (staged to manage runtime and memory pressure), e.g.:

```
python main.py maintenance reindex --collection exp_full_cs256_co64 --operation reembed --no-backup
```
