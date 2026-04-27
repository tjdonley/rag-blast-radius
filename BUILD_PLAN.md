# rag-blast-radius Build Plan

## 1. Project Thesis

`rag-blast-radius` is an OSS CLI for pre-deploy safety checks on RAG changes.

It helps engineers answer:

> What indexes, caches, evals, and rollout steps become unsafe if I change embeddings, chunking, retrieval, reranking, or cache namespaces?

The first version should focus on manifest diffing and deterministic blast-radius analysis, not monitoring, hosted services, dashboards, or enterprise governance.

## 2. Recommended Name

Repository:

```text
rag-blast-radius
```

CLI:

```text
rag-blast
```

## 3. Core Scope

V0 should include:

- A `.rag-manifest.json` schema
- A CLI for initializing and checking manifests
- A deterministic diff engine
- A rules engine that maps changes to risks
- Human-readable blast-radius reports
- JSON output for CI usage
- Examples that teach common RAG migration risks
- Tests for schema validation, diffing, and rules

V0 should not include:

- Hosted service
- Monitoring
- Web dashboard
- Enterprise audit workflows
- Automatic full-repo scanning
- Cost modeling
- Broad vector database support
- Multiple framework integrations

## 4. Technical Stack

Use Python for the initial implementation.

Recommended tools:

- `uv` for packaging and dependency management
- `typer` for the CLI
- `pydantic` for manifest schema validation
- `rich` for terminal output
- `pytest` for tests
- `ruff` for linting and formatting

## 5. Repository Structure

```text
rag-blast-radius/
  README.md
  BUILD_PLAN.md
  pyproject.toml
  src/
    rag_blast/
      __init__.py
      cli.py
      manifest.py
      diff.py
      rules.py
      report.py
  tests/
    test_manifest.py
    test_diff.py
    test_rules.py
  examples/
    openai_ada_to_3_large/
      old.json
      new.json
    chunk_size_change/
      old.json
      new.json
    reranker_added/
      old.json
      new.json
    semantic_cache_namespace_bug/
      old.json
      new.json
```

## 6. Phase 1: Repository And CLI Skeleton

Goal: create a working Python CLI package.

Deliverables:

- Public GitHub repo named `rag-blast-radius`
- Python package under `src/rag_blast`
- CLI entrypoint named `rag-blast`
- Basic commands: `rag-blast --help`, `rag-blast init`, `rag-blast check --old old.json --new new.json`, and `rag-blast explain RULE_ID`

Acceptance criteria:

- Package installs locally
- CLI command runs
- Tests run successfully
- README explains the basic purpose in under 30 seconds

## 7. Phase 2: Manifest Schema

Goal: define the core artifact that the project revolves around.

Initial manifest fields:

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

Deliverables:

- Pydantic models for the manifest
- Validation errors that are understandable
- `rag-blast init` creates a starter manifest
- Example manifests under `examples/`

Acceptance criteria:

- Invalid manifests fail clearly
- Valid manifests parse consistently
- Schema is simple enough for users to write manually

## 8. Phase 3: Manifest Diff Engine

Goal: compare two manifests and identify meaningful RAG changes.

Detected change categories:

- Embedding provider changed
- Embedding model changed
- Embedding dimensions changed
- Chunking strategy changed
- Chunk size changed
- Chunk overlap changed
- Vector store provider changed
- Vector collection changed
- Retriever `top_k` changed
- Hybrid retrieval changed
- Reranker added, removed, or changed
- Semantic cache namespace unchanged after embedding changes
- Eval dataset missing or changed

Deliverables:

- Structured diff result object
- Human-readable changed-field output
- Tests for all major field changes

Acceptance criteria:

- Diff output is deterministic
- Nested field changes are easy to inspect
- Diff engine does not depend on vector DBs or RAG frameworks

## 9. Phase 4: Rules Engine

Goal: convert detected changes into risk findings and recommendations.

Initial rules:

```text
REEMBED_REQUIRED
VECTOR_INDEX_INCOMPATIBLE
SEMANTIC_CACHE_UNSAFE
RETRIEVAL_BASELINE_STALE
CHUNKING_CHANGED
RERANKER_CHANGED
RETRIEVER_BEHAVIOR_CHANGED
SHADOW_INDEX_RECOMMENDED
ROLLBACK_REQUIRES_OLD_INDEX
```

Severity levels:

```text
LOW
MEDIUM
HIGH
```

Example rule behavior:

- If embedding model changes, trigger `REEMBED_REQUIRED`, `VECTOR_INDEX_INCOMPATIBLE`, `SEMANTIC_CACHE_UNSAFE`, `RETRIEVAL_BASELINE_STALE`, and `SHADOW_INDEX_RECOMMENDED`

Deliverables:

- Rules implemented as code first
- Rule descriptions
- Rule recommendations
- `rag-blast explain RULE_ID`

Acceptance criteria:

- Each rule has tests
- Same input always produces same output
- Rule output is understandable without reading source code

## 10. Phase 5: Reports

Goal: make risk legible in terminal and CI.

CLI output should look like:

```text
RAG BLAST RADIUS REPORT

Risk: HIGH

Detected changes:
  - embedding.model changed
  - embedding.dimensions changed

Invalidation rules triggered:
  - REEMBED_REQUIRED
  - VECTOR_INDEX_INCOMPATIBLE
  - SEMANTIC_CACHE_UNSAFE
  - RETRIEVAL_BASELINE_STALE
  - SHADOW_INDEX_RECOMMENDED

Recommended rollout:
  1. Build shadow index.
  2. Replay representative production queries.
  3. Compare retrieval overlap and answer quality.
  4. Canary by tenant or traffic percentage.
  5. Keep old index and cache namespace until rollback window closes.
```

Output formats:

- Default human-readable terminal report
- `--format json` for CI and automation

Deliverables:

- Terminal report renderer
- JSON report renderer
- Exit codes based on severity threshold

Acceptance criteria:

- Humans can understand the report quickly
- Machines can parse the JSON output
- CI can fail on high-risk changes

## 11. Phase 6: Examples

Goal: make the project educational and easy to evaluate.

Initial examples:

```text
examples/
  openai_ada_to_3_large/
  chunk_size_change/
  reranker_added/
  semantic_cache_namespace_bug/
  weaviate_alias_migration/
  qdrant_dual_collection_migration/
```

Each example should include:

- `old.json`
- `new.json`
- Expected report summary
- Short README explaining the risk

Acceptance criteria:

- A new user can understand the project by reading examples
- Examples double as regression fixtures
- Examples cover common production RAG mistakes

## 12. Phase 7: GitHub Action

Goal: make `rag-blast-radius` useful in PR workflows.

Example usage:

```yaml
- name: Check RAG blast radius
  uses: yourname/rag-blast-radius-action@v0
  with:
    old_manifest: rag-manifest.prod.json
    new_manifest: rag-manifest.pr.json
    fail_on: high
```

Expected output:

```text
Blocking deploy:
  HIGH: embedding model changed without shadow index plan
  HIGH: semantic cache namespace unchanged after embedding change
  MEDIUM: retrieval evals not configured
```

Deliverables:

- GitHub Action wrapper
- Action README
- `fail_on` severity option
- CI example in main README

Acceptance criteria:

- Action can run in a sample workflow
- High-risk changes can block a PR
- JSON output remains available for advanced users

## 13. Phase 8: First Integration

Goal: add one narrow real-world integration after the core CLI is useful.

Recommended first integration:

```text
LlamaIndex + Qdrant
```

Reason:

- Python users are likely to use LlamaIndex
- Qdrant has clear migration concepts
- Integration can stay focused on manifest generation or enrichment

Do not perform migrations in v1.

Integration should only:

- Inspect known configuration patterns
- Generate a partial manifest
- Warn when required fields must be filled manually

Acceptance criteria:

- Integration is useful but not magical
- Missing data is surfaced clearly
- No false confidence from incomplete scanning

## 14. README Structure

README should include:

- One-line project description
- Why RAG changes are risky
- Quickstart
- Example manifest
- Example `check` output
- Rule list
- GitHub Action usage
- Non-goals
- Roadmap

Opening copy:

```md
# rag-blast-radius

Pre-deploy safety checks for RAG changes.

RAG systems are full of hidden state: document embeddings, vector indexes, semantic caches, chunking configs, retriever settings, rerankers, eval baselines, and prompt assumptions.

Changing one component can silently invalidate the others.

`rag-blast-radius` diffs RAG manifests and reports what becomes unsafe before you deploy.
```

## 15. Success Metrics

Do not optimize for stars first.

Better signals:

- Someone runs it on a real RAG repo
- Someone discovers a stale cache, index, or eval baseline
- Someone adds it to CI
- Someone opens an issue asking for a framework or vector DB integration
- Someone proposes additions to the manifest schema

Best early validation:

```text
A PR adopts rag-blast-radius as a required pre-deploy check.
```

## 16. Future Commercial Paths

Possible future product directions if OSS usage validates demand:

- Hosted manifest history
- Signed release reports
- Approval workflows
- SSO
- Team policies
- Integration with Langfuse, LangSmith, Braintrust, Phoenix, Arize, Qdrant, Weaviate, and LlamaIndex
- Organization-wide RAG change inventory
- Policy-as-code for RAG deployment safety

Do not build these until usage pulls the project there.

## 17. Build Order Summary

Recommended implementation order:

1. Create repo and package skeleton
2. Add CLI entrypoint
3. Implement manifest schema
4. Implement manifest diffing
5. Implement deterministic rules
6. Add terminal report output
7. Add JSON output
8. Add examples
9. Add tests
10. Improve README
11. Add GitHub Action
12. Add one narrow integration

## 18. Initial Release Definition

The first useful release is ready when this works:

```bash
rag-blast check --old examples/openai_ada_to_3_large/old.json --new examples/openai_ada_to_3_large/new.json
```

And produces:

```text
Risk: HIGH

Detected changes:
  - embedding.model changed
  - embedding.dimensions changed

Invalidation rules triggered:
  - REEMBED_REQUIRED
  - VECTOR_INDEX_INCOMPATIBLE
  - SEMANTIC_CACHE_UNSAFE
  - RETRIEVAL_BASELINE_STALE
  - SHADOW_INDEX_RECOMMENDED
```

At that point, the project is useful enough to share publicly.
