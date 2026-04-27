import json
from pathlib import Path

import pytest

from rag_blast.manifest import (
    ManifestLoadError,
    load_manifest,
    starter_manifest,
    validate_manifest,
    write_starter_manifest,
)


def test_starter_manifest_contains_core_sections() -> None:
    manifest = starter_manifest()

    assert manifest["app"] == "customer-support-rag"
    assert manifest["embedding"]["model"] == "text-embedding-ada-002"
    assert manifest["vector_store"]["provider"] == "qdrant"


def test_write_and_load_starter_manifest(tmp_path) -> None:
    path = tmp_path / ".rag-manifest.json"

    write_starter_manifest(path)

    assert load_manifest(path) == starter_manifest()


def test_load_manifest_rejects_non_object_json(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ManifestLoadError):
        load_manifest(path)


def test_validate_manifest_adds_default_lists_and_optional_fields() -> None:
    manifest = validate_manifest(
        {
            "app": "customer-support-rag",
            "environment": "prod",
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
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
            },
        }
    )

    assert manifest["caches"] == []
    assert manifest["evals"] == []
    assert manifest["retriever"]["reranker"] is None
    assert manifest["vector_store"]["alias"] is None


def test_validate_manifest_rejects_missing_required_section() -> None:
    manifest = starter_manifest()
    del manifest["embedding"]

    with pytest.raises(ManifestLoadError, match="embedding: Field required"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_invalid_dimensions() -> None:
    manifest = starter_manifest()
    manifest["embedding"]["dimensions"] = 0

    with pytest.raises(ManifestLoadError, match="embedding.dimensions"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_coerced_numeric_types() -> None:
    manifest = starter_manifest()
    manifest["embedding"]["dimensions"] = "1536"

    with pytest.raises(ManifestLoadError, match="embedding.dimensions"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_coerced_boolean_types() -> None:
    manifest = starter_manifest()
    manifest["retriever"]["hybrid"] = "false"

    with pytest.raises(ManifestLoadError, match="retriever.hybrid"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_invalid_chunk_overlap() -> None:
    manifest = starter_manifest()
    manifest["chunking"]["chunk_overlap"] = manifest["chunking"]["chunk_size"]

    with pytest.raises(ManifestLoadError, match="chunk_overlap must be smaller"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_extra_keys() -> None:
    manifest = starter_manifest()
    manifest["embedding"]["extra"] = "typo"

    with pytest.raises(ManifestLoadError, match="embedding.extra"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_string_reranker() -> None:
    manifest = starter_manifest()
    manifest["retriever"]["reranker"] = "cohere/rerank-english-v3.0"

    with pytest.raises(ManifestLoadError, match="retriever.reranker"):
        validate_manifest(manifest)


def test_example_manifests_are_valid() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"
    manifest_paths = sorted(examples_dir.glob("**/*.json"))

    assert manifest_paths
    for path in manifest_paths:
        load_manifest(path)
