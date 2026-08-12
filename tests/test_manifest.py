import json
from pathlib import Path

import jsonschema
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


def test_manifest_json_schema_is_a_valid_schema_document() -> None:
    jsonschema.Draft202012Validator.check_schema(manifest_json_schema())


def test_manifest_json_schema_accepts_a_starter_manifest_with_a_schema_reference() -> None:
    manifest = starter_manifest()
    manifest["$schema"] = "./.rag-manifest.schema.json"

    jsonschema.validate(manifest, manifest_json_schema())


def test_manifest_json_schema_rejects_unknown_keys() -> None:
    manifest = starter_manifest()
    manifest["nonsense"] = 1

    with pytest.raises(jsonschema.ValidationError, match="Additional properties"):
        jsonschema.validate(manifest, manifest_json_schema())


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda m: m.__setitem__("app", "   "), id="top-level-string"),
        pytest.param(lambda m: m["embedding"].__setitem__("model", "  "), id="nested-string"),
        pytest.param(lambda m: m["vector_store"].__setitem__("alias", " "), id="nullable-string"),
        pytest.param(lambda m: m["caches"][0].__setitem__("namespace", "\t"), id="list-item"),
        pytest.param(lambda m: m["evals"][0].__setitem__("name", " "), id="list-item-name"),
    ],
)
def test_exported_schema_rejects_whitespace_only_strings_like_the_runtime_does(mutate) -> None:
    """NonEmptyString strips before checking length, so minLength alone under-reports."""
    manifest = starter_manifest()
    mutate(manifest)

    with pytest.raises(ManifestLoadError):
        validate_manifest(manifest)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, manifest_json_schema())


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda m: m["embedding"].__setitem__("dimensions", 0), id="positive-int"),
        pytest.param(
            lambda m: m["embedding"].__setitem__("dimensions", "1536"), id="int-as-string"
        ),
        pytest.param(lambda m: m["retriever"].__setitem__("top_k", -1), id="negative-top-k"),
        pytest.param(
            lambda m: m["chunking"].__setitem__("chunk_overlap", -5), id="negative-overlap"
        ),
        pytest.param(lambda m: m["retriever"].__setitem__("hybrid", "yes"), id="bool-as-string"),
        pytest.param(lambda m: m.pop("embedding"), id="missing-section"),
    ],
)
def test_exported_schema_agrees_with_the_runtime_validator(mutate) -> None:
    manifest = starter_manifest()
    mutate(manifest)

    with pytest.raises(ManifestLoadError):
        validate_manifest(manifest)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, manifest_json_schema())


def test_cross_field_chunking_rule_is_documented_as_runtime_only() -> None:
    """JSON Schema cannot compare two fields, so the schema must say where that rule lives."""
    manifest = starter_manifest()
    manifest["chunking"]["chunk_overlap"] = manifest["chunking"]["chunk_size"]

    with pytest.raises(ManifestLoadError, match="chunk_overlap must be smaller than chunk_size"):
        validate_manifest(manifest)

    schema = manifest_json_schema()
    jsonschema.validate(manifest, schema)
    assert "chunk_overlap < chunking.chunk_size" in schema["description"]
