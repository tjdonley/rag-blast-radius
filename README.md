# rag-blast-radius

Pre-deploy safety checks for RAG changes.

RAG systems accumulate hidden state: document embeddings, vector indexes, semantic caches, chunking configs, retriever settings, rerankers, eval baselines, and prompt assumptions.

Changing one component can silently invalidate the others. `rag-blast-radius` is an OSS CLI that makes those dependencies explicit before deployment.

## Status

The CLI includes package wiring, starter manifest generation, typed manifest validation, categorized manifest diffing, deterministic risk rules, CI-friendly reports, rule explanations, examples, and tests.

The GitHub Action and integrations are planned in `BUILD_PLAN.md`.

## Install Locally

```bash
uv sync
uv run rag-blast --help
```

If you are not using `uv`, install the package in editable mode:

```bash
python -m pip install -e .
rag-blast --help
```

## Quickstart

Create a starter manifest:

```bash
rag-blast init
```

Compare two manifests:

```bash
rag-blast check --old old.json --new new.json
```

Emit machine-readable JSON:

```bash
rag-blast check --old old.json --new new.json --format json
```

Fail CI on high-risk or unassessed changes:

```bash
rag-blast check --old old.json --new new.json --fail-on high
```

From a repo checkout, try one of the included examples:

```bash
rag-blast check --old examples/openai_ada_to_3_large/old.json --new examples/openai_ada_to_3_large/new.json
```

Explain a rule:

```bash
rag-blast explain REEMBED_REQUIRED
```

## Manifest Schema

`rag-blast-radius` centers on a RAG manifest. The default filename is `.rag-manifest.json`.

```json
{
  "app": "customer-support-rag",
  "environment": "prod",
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "dimensions": 1536
  },
  "chunking": {
    "strategy": "recursive_character",
    "chunk_size": 800,
    "chunk_overlap": 100
  },
  "vector_store": {
    "provider": "qdrant",
    "collection": "support_docs_v3"
  },
  "retriever": {
    "top_k": 8,
    "hybrid": false,
    "reranker": null
  },
  "caches": [
    {
      "type": "semantic_cache",
      "namespace": "support_rag_prod_v4",
      "embedding_model": "text-embedding-ada-002"
    }
  ],
  "evals": [
    {
      "name": "retrieval_golden",
      "path": "evals/retrieval_golden.jsonl"
    }
  ]
}
```

Validation catches missing required sections, empty strings, invalid numeric values, stringified numbers or booleans, chunk overlaps that are not smaller than chunk size, and unknown keys that are likely typos.

`retriever.reranker` can be `null` or an object with a required `model` and optional `provider`.

## Diff Output

`rag-blast check` returns deterministic, categorized manifest changes and triggered risk rules.

Example JSON output includes paths, categories, summaries, old/new values, rule findings, recommended rollout steps, and the highest triggered severity. This excerpt shows the shape:

```json
{
  "risk": "HIGH",
  "change_count": 2,
  "categories": [
    "embedding_model_changed",
    "semantic_cache_namespace_unchanged"
  ],
  "changes": [
    {
      "path": "caches[support_rag_prod_v4].namespace",
      "category": "semantic_cache_namespace_unchanged",
      "summary": "Semantic cache namespace unchanged after embedding, chunking, or retrieval change",
      "old": "support_rag_prod_v4",
      "new": "support_rag_prod_v4"
    },
    {
      "path": "embedding.model",
      "category": "embedding_model_changed",
      "summary": "Embedding model changed",
      "old": "text-embedding-ada-002",
      "new": "text-embedding-3-large"
    }
  ],
  "finding_count": 6,
  "findings": [
    {
      "rule_id": "REEMBED_REQUIRED",
      "severity": "HIGH",
      "summary": "Embedding provider, model, dimensions, or chunking changed.",
      "recommendation": "Rebuild document embeddings before serving traffic from the changed embedding or chunking configuration.",
      "change_paths": ["embedding.model"]
    }
  ],
  "recommended_rollout": [
    "Regenerate document embeddings for the proposed manifest."
  ],
  "note": "Additional findings omitted from this excerpt."
}
```

`--fail-on` accepts `none`, `low`, `medium`, or `high`. The default is `none`. If changes are present but no rule can assess them, the report risk is `UNASSESSED`; any enabled threshold fails that report so unknown-impact changes do not silently pass CI.

## Rules

Initial deterministic rules:

- `REEMBED_REQUIRED`
- `VECTOR_INDEX_INCOMPATIBLE`
- `SEMANTIC_CACHE_UNSAFE`
- `RETRIEVAL_BASELINE_STALE`
- `CHUNKING_CHANGED`
- `RERANKER_CHANGED`
- `RETRIEVER_BEHAVIOR_CHANGED`
- `SHADOW_INDEX_RECOMMENDED`
- `ROLLBACK_REQUIRES_OLD_INDEX`

Explain a rule locally:

```bash
rag-blast explain REEMBED_REQUIRED
```

## Repository

The public repository name is `rag-blast-radius` and the CLI command is `rag-blast`.
