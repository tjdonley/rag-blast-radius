# Reranker Added

This example adds a Cohere reranker and increases retriever `top_k` from `8` to `12`.

The vector index does not need to be rebuilt, but retrieval behavior changes enough that retrieval overlap and answer-quality evals should be replayed.

Expected outcome:

- Risk: `MEDIUM`
- Reranker behavior changed
- Retriever behavior changed
- Retrieval baselines are stale

Run it:

```bash
uv run rag-blast check --old examples/reranker_added/old.json --new examples/reranker_added/new.json
```
