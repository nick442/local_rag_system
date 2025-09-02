# Retrieval

The retriever (`src/retriever.py`) exposes a simple API to fetch the most relevant chunks and assemble an LLM‑ready context.

## Methods
- Vector: cosine similarity search over embeddings.
- Keyword: SQLite FTS5 full‑text search with BM25 ranking.
- Hybrid: weighted mix of vector and keyword (e.g., `alpha=0.7`).

Backends are provided by `src/vector_database.py` via `create_vector_index`. Embeddings come from `src/embedding_service.py`.

## Flow
1. Embed query with `EmbeddingService`.
2. Search `VectorDatabase` (vector/keyword/hybrid) with optional `collection_id`.
3. Wrap results as `RetrievalResult` objects.
4. Optionally expand with neighboring chunks for context.
5. Assemble a token‑budgeted context string.

## Context Assembly
- Deduplicates repeated content.
- Includes lightweight source lines like `Source: <file> | Chunk: N | Score: s`.
- Respects `max_context_tokens` using the configured tokenizer.

## Tuning Knobs
- `k`: number of base results.
- `method`: `vector`, `keyword`, `hybrid`.
- `max_context_tokens`: prompt budget for context assembly.
- `chunk_size` / `chunk_overlap`: set at ingestion to control granularity.

## Metadata Filtering
Call `filter_by_metadata(results, filters)` to constrain matches by attributes (e.g., page ranges or tags). Supports equality, `$in`, and regex‑like matching.

## Notes
- If `sqlite-vec` is present, vector search is fast and memory‑efficient. Fallbacks preserve correctness without the extension.
- Hybrid retrieval is robust when queries contain both topical and lexical signals.

