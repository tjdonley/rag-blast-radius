# Chunk Size Change

This example changes chunk size from `800` to `1200` and overlap from `100` to `150`.

Changing chunking changes the document units that get embedded, so the old vectors and retrieval baselines no longer describe the proposed system.

Expected outcome:

- Risk: `HIGH`
- Chunks and document embeddings must be regenerated
- Retrieval evals should be replayed
- Shadow-index and rollback planning are recommended

Run it:

```bash
uv run rag-blast check --old examples/chunk_size_change/old.json --new examples/chunk_size_change/new.json
```
