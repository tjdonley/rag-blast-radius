# Qdrant Dual Collection Migration

This example models a Qdrant dual-collection cutover where the embedding and chunking configuration stay the same, but the active collection changes from `support_docs_v1` to `support_docs_v2`.

This is lower risk than an embedding migration, but it still needs a shadow-read or canary plan and a rollback window.

Expected outcome:

- Risk: `MEDIUM`
- Shadow index is recommended
- Rollback requires preserving the old collection

Run it:

```bash
uv run rag-blast check --old examples/qdrant_dual_collection_migration/old.json --new examples/qdrant_dual_collection_migration/new.json
```
