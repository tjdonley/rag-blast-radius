# rag-blast-radius

Pre-deploy safety checks for RAG changes.

RAG systems accumulate hidden state: document embeddings, vector indexes, semantic caches, chunking configs, retriever settings, rerankers, eval baselines, and prompt assumptions.

Changing one component can silently invalidate the others. `rag-blast-radius` is an OSS CLI that makes those dependencies explicit before deployment.

## Status

The CLI includes package wiring, starter manifest generation, typed manifest validation, basic manifest diffing, rule explanations, examples, and tests.

The deterministic risk rules, richer reports, GitHub Action, and integrations are planned in `BUILD_PLAN.md`.

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

## Repository

The public repository name is `rag-blast-radius` and the CLI command is `rag-blast`.
