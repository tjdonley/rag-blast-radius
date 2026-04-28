<p align="center">
  <img src="docs/assets/rag-blast-logo.png" alt="rag-blast-radius logo" width="170">
</p>

<h1 align="center">rag-blast-radius</h1>

<p align="center">
  Pre-deploy safety checks for RAG changes.
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-19c2ff">
  <img alt="CLI rag-blast" src="https://img.shields.io/badge/CLI-rag--blast-536dff">
  <img alt="Status early OSS" src="https://img.shields.io/badge/status-early%20OSS-72e6ae">
</p>

RAG systems are full of hidden state: document embeddings, vector indexes, semantic caches, chunking configs, retriever settings, rerankers, eval baselines, and prompt assumptions.

Changing one component can silently invalidate the others. `rag-blast-radius` diffs RAG manifests and reports what becomes unsafe before you deploy.

<p align="center">
  <img src="docs/assets/blast-radius-map.svg" alt="A map of RAG configuration changes and affected production assets">
</p>

## Why RAG Changes Are Risky

RAG deployments often couple independently managed systems: an embedding model, a vector collection, chunking code, retriever settings, rerankers, semantic caches, and eval baselines. A small code or config change can make existing vectors incomparable, leave caches keyed to stale embeddings, or make retrieval evals no longer describe production behavior.

`rag-blast-radius` makes those dependencies explicit by comparing two manifests and applying deterministic local rules. It does not call hosted services or inspect production data.

## What It Helps Answer

- Did the embedding model, dimensions, or provider change?
- Did chunking change in a way that requires regenerated embeddings?
- Did a vector collection, semantic cache namespace, or eval baseline drift?
- Can a reviewer understand the RAG deployment risk without reading app code?
- Can CI get a stable JSON summary of the manifest diff?

## Status

The CLI includes package wiring, starter manifest generation, typed manifest validation, categorized manifest diffing, deterministic risk rules, CI-friendly reports, a GitHub Action wrapper, a narrow LlamaIndex + Qdrant integration, rule explanations, examples, and tests.

## Install Locally

```bash
uv sync
uv run rag-blast --help
```

Without `uv`:

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

## How It Works

<p align="center">
  <img src="docs/assets/check-flow.svg" alt="RAG manifests flow through validation, diffing, reporting, and CI">
</p>

1. Write down the current and proposed RAG state as manifests.
2. Validate both manifests with strict, typed schema checks.
3. Diff the manifests into stable paths such as `embedding.model`.
4. Apply deterministic rules that map changes to risk and rollout actions.
5. Render the result as readable text or JSON for CI and pull requests.

## Example Check Output

The example command above produces a high-risk report with required rollout steps. This excerpt omits additional cache changes, unassessed paths, and rollout lines:

```text
RAG BLAST RADIUS REPORT

Risk: HIGH

Detected changes:
  - embedding.dimensions (embedding_dimensions_changed): Embedding dimensions changed; 1536 -> 3072
  - embedding.model (embedding_model_changed): Embedding model changed; text-embedding-ada-002 -> text-embedding-3-large
  - vector_store.collection (vector_collection_changed): Vector collection changed; support_docs_v3 -> support_docs_v4

Invalidation rules triggered:
  - HIGH: REEMBED_REQUIRED - Embedding provider, model, dimensions, or chunking changed.
  - HIGH: VECTOR_INDEX_INCOMPATIBLE - Existing vectors may not be comparable to new query vectors.
  - MEDIUM: RETRIEVAL_BASELINE_STALE - Retrieval eval baselines may no longer describe the proposed system.

Recommended rollout:
  1. Regenerate document embeddings for the proposed manifest.
  2. Build a shadow vector index before serving new query embeddings.
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

## Non-Goals

- Hosted service
- Monitoring or online evaluation
- Web dashboard
- Enterprise approval workflows
- Automatic full-repo scanning
- Automatic migrations
- Cost modeling
- Broad vector database support in v0

## Roadmap

- Keep the manifest schema simple enough to write and review manually.
- Expand deterministic rules only when they map to clear rollout actions.
- Add integrations when real users ask for specific frameworks or vector databases.
- Prefer local, inspectable outputs over hosted workflows until usage proves a stronger need.
- Consider signed release reports, manifest history, and policy-as-code after the CLI has real adoption.
