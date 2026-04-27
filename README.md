# rag-blast-radius

Pre-deploy safety checks for RAG changes.

RAG systems accumulate hidden state: document embeddings, vector indexes, semantic caches, chunking configs, retriever settings, rerankers, eval baselines, and prompt assumptions.

Changing one component can silently invalidate the others. `rag-blast-radius` is an OSS CLI that makes those dependencies explicit before deployment.

## Status

Phase 1 is a working CLI skeleton. It includes package wiring, starter manifest generation, basic manifest diffing, rule explanations, and tests.

The deeper manifest schema, deterministic risk rules, rich reports, GitHub Action, and integrations are planned in `BUILD_PLAN.md`.

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

Explain a rule:

```bash
rag-blast explain REEMBED_REQUIRED
```

## Repository

The intended public repository name is `rag-blast-radius` and the CLI command is `rag-blast`.
