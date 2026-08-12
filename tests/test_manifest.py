import json
from pathlib import Path

import pytest

from rag_blast.diff import diff_manifests
from rag_blast.manifest import (
    ManifestLoadError,
    load_manifest,
    manifest_json_schema,
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
    manifest_paths = sorted(examples_dir.glob("*/old.json")) + sorted(
        examples_dir.glob("*/new.json")
    )

    assert manifest_paths
    for path in manifest_paths:
        load_manifest(path)


def test_validate_manifest_accepts_and_drops_a_schema_reference() -> None:
    manifest = starter_manifest()
    manifest["$schema"] = "./rag-manifest.schema.json"

    validated = validate_manifest(manifest)

    assert "$schema" not in validated
    assert validated == starter_manifest()


def test_schema_reference_never_appears_as_a_manifest_change() -> None:
    old = starter_manifest()
    new = starter_manifest()
    old["$schema"] = "./rag-manifest.schema.json"
    new["$schema"] = "https://example.com/other.schema.json"

    diff = diff_manifests(validate_manifest(old), validate_manifest(new))

    assert diff.changes == ()


def test_validate_manifest_rejects_a_non_string_schema_reference() -> None:
    manifest = starter_manifest()
    manifest["$schema"] = 42

    with pytest.raises(ManifestLoadError, match=r"\$schema: Input should be a valid string"):
        validate_manifest(manifest)


def test_manifest_json_schema_declares_the_schema_reference_property() -> None:
    schema = manifest_json_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "RAG Manifest"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["$schema"]["type"] == "string"
    assert set(schema["required"]) == {
        "app",
        "environment",
        "embedding",
        "chunking",
        "vector_store",
        "retriever",
    }


def test_manifest_json_schema_describes_every_manifest_section() -> None:
    properties = manifest_json_schema()["properties"]

    for section in ("app", "environment", "embedding", "chunking", "vector_store", "retriever"):
        assert section in properties
    assert set(manifest_json_schema()["$defs"]) >= {
        "EmbeddingConfig",
        "ChunkingConfig",
        "VectorStoreConfig",
        "RetrieverConfig",
        "CacheConfig",
        "EvalConfig",
    }
