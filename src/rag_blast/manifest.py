from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestLoadError(Exception):
    """Raised when a manifest cannot be loaded as a JSON object."""


def starter_manifest() -> dict[str, Any]:
    """Return a starter RAG manifest users can edit."""
    return {
        "app": "customer-support-rag",
        "environment": "prod",
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-ada-002",
            "dimensions": 1536,
        },
        "chunking": {
            "strategy": "recursive_character",
            "chunk_size": 800,
            "chunk_overlap": 100,
        },
        "vector_store": {
            "provider": "qdrant",
            "collection": "support_docs_v3",
        },
        "retriever": {
            "top_k": 8,
            "hybrid": False,
            "reranker": None,
        },
        "caches": [
            {
                "type": "semantic_cache",
                "namespace": "support_rag_prod_v4",
                "embedding_model": "text-embedding-ada-002",
            }
        ],
        "evals": [
            {
                "name": "retrieval_golden",
                "path": "evals/retrieval_golden.jsonl",
            }
        ],
    }


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a manifest from disk and ensure it is a JSON object."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestLoadError(f"Unable to read manifest {path}: {error}") from error

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ManifestLoadError(f"Invalid JSON in manifest {path}: {error.msg}") from error

    if not isinstance(data, dict):
        raise ManifestLoadError(f"Manifest must be a JSON object: {path}")

    return data


def write_starter_manifest(path: Path, *, force: bool = False) -> None:
    """Write a starter manifest to disk."""
    if path.exists() and not force:
        raise FileExistsError(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(starter_manifest(), indent=2) + "\n", encoding="utf-8")
