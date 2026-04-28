# Weaviate Alias Migration

This example models a Weaviate collection migration where the app-facing alias remains `SupportDocs`, but the backing collection changes from `SupportDocs_v3` to `SupportDocs_v4`.

The embedding model also changes, so the alias cutover still needs a shadow index and rollback plan.

Expected outcome:

- Risk: `HIGH`
- Re-embedding is required
- Existing vectors are incompatible with the proposed query embeddings
- The old collection should be preserved for rollback

Run it:

```bash
uv run rag-blast check --old examples/weaviate_alias_migration/old.json --new examples/weaviate_alias_migration/new.json
```
