# Full-Scale Run Plan — realistic_full_production

This document outlines the staged plan to run the 20‑configuration chunking template against the `realistic_full_production` base, using derived collections prefixed by `exp_full_cs`.

## Stage 1: Clone (done)
- exp_full_cs256_co51, exp_full_cs256_co64, exp_full_cs512_co128 cloned from realistic_full_production.
- See README.md for counts.

## Stage 2: Re-embed (in progress)
- Re-embed exp_full_cs256_co64 and exp_full_cs512_co128 (large jobs).
- Command (idempotent):
```
python main.py maintenance reindex --collection exp_full_cs256_co64 --operation reembed --no-backup
python main.py maintenance reindex --collection exp_full_cs512_co128 --operation reembed --no-backup
```

## Stage 3: Clone remaining configs (batches)
- For each (chunk_size, overlap):
```
python main.py collection clone realistic_full_production \
  --target exp_full_cs{CHUNK}_co{OVERLAP} \
  --chunk-size {CHUNK} --chunk-overlap {OVERLAP} --no-embed
```
- Suggested order: start with 256/512 families, then 128/768/1024.

## Stage 4: Re-embed remaining configs (batches)
- As above, use maintenance reindex reembed per target collection.

## Stage 5: Run template (20 configs × 52 queries)
- Use collection prefix to ensure derived collections resolve to `exp_full_cs...` IDs:
```
python main.py experiment template chunk_optimization \
  --queries test_data/enhanced_evaluation_queries.json \
  --collection-prefix exp_full_cs \
  --output experiments/experiment_1_v2_full_run_YYYYMMDD_HHMMSS/results.json
```

## Notes
- This will take many hours; recommend running overnight and verifying logs periodically.
- If any target collection is missing embeddings, retrieval will fallback poorly. Ensure re-embed completed for all before starting the template.
