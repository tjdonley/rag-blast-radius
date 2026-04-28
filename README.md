# rag-blast-radius

Pre-deploy safety checks for RAG changes.

RAG systems accumulate hidden state: document embeddings, vector indexes, semantic caches, chunking configs, retriever settings, rerankers, eval baselines, and prompt assumptions.

Changing one component can silently invalidate the others. `rag-blast-radius` is an OSS CLI that makes those dependencies explicit before deployment.

## Status

The CLI includes package wiring, starter manifest generation, typed manifest validation, categorized manifest diffing, deterministic risk rules, CI-friendly reports, a GitHub Action wrapper, a narrow LlamaIndex + Qdrant integration, rule explanations, examples, and tests.

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

See `examples/README.md` for the full catalog of migration scenarios and expected report summaries.

Generate a partial manifest draft from a LlamaIndex + Qdrant Python config:

```bash
rag-blast integrations llamaindex-qdrant --source src/rag_app.py --output .rag-manifest.partial.json
```

The integration only inspects known local configuration patterns. It prints `Manual review required` warnings for required manifest fields that must be filled manually before the draft is used with `rag-blast check`.

See `docs/llamaindex-qdrant.md` for supported patterns, limitations, and example output.

Explain a rule:

```bash
rag-blast explain REEMBED_REQUIRED
```

## GitHub Action

Use the action in pull requests to block risky RAG manifest changes before deployment:

```yaml
name: RAG Blast Radius

on:
  pull_request:

jobs:
  rag-blast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check RAG blast radius
        id: rag_blast
        uses: tjdonley/rag-blast-radius@v0
        with:
          old_manifest: manifests/rag-prod.json
          new_manifest: .rag-manifest.json
          fail_on: high
```

The action prints the normal text report, writes a job summary, and exposes `risk`, `change_count`, `finding_count`, and `unassessed_change_count` outputs. Set `format: json` when workflow automation needs the raw JSON report in the logs.

See `docs/github-action.md` for all action inputs, outputs, and a JSON workflow example.

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
