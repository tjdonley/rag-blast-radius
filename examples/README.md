# Examples

These examples are small RAG manifest diffs that double as regression fixtures.

Run any example from a repo checkout:

```bash
uv run rag-blast check --old examples/openai_ada_to_3_large/old.json --new examples/openai_ada_to_3_large/new.json
```

Each directory contains:

- `old.json`: baseline manifest
- `new.json`: proposed manifest
- `expected-summary.json`: stable report fields asserted by tests
- `README.md`: short explanation of the production risk

## Catalog

- `openai_ada_to_3_large`: embedding model and dimension migration with a new vector collection and cache namespace.
- `chunk_size_change`: chunk size and overlap change requiring chunk regeneration and index rebuild.
- `reranker_added`: reranker addition plus retriever `top_k` change.
- `semantic_cache_namespace_bug`: embedding migration that accidentally reuses a semantic cache namespace.
- `weaviate_alias_migration`: Weaviate collection migration behind a stable alias.
- `qdrant_dual_collection_migration`: Qdrant dual-collection cutover with the same embedding model.
