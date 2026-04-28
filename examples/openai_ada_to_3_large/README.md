# OpenAI Ada To 3 Large

This example models an embedding migration from `text-embedding-ada-002` to `text-embedding-3-large`.

The embedding dimensions change from `1536` to `3072`, and the proposed manifest points at a new vector collection and cache namespace.

Expected outcome:

- Risk: `HIGH`
- Re-embedding is required
- Existing vectors are not comparable to new query vectors
- A shadow index and rollback window are recommended
- Cache add/remove paths remain unassessed until cache-specific rules are expanded

Run it:

```bash
uv run rag-blast check --old examples/openai_ada_to_3_large/old.json --new examples/openai_ada_to_3_large/new.json
```
