# Semantic Cache Namespace Bug

This example changes embedding model and dimensions but accidentally keeps the same semantic cache namespace.

That is unsafe because cache entries may have been produced with the old embedding or retrieval behavior while the proposed system is using a new one.

Expected outcome:

- Risk: `HIGH`
- Semantic cache is unsafe
- Existing vectors are incompatible with the new query embeddings
- The cache namespace should change before rollout

Run it:

```bash
uv run rag-blast check --old examples/semantic_cache_namespace_bug/old.json --new examples/semantic_cache_namespace_bug/new.json
```
