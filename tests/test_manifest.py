import json

import pytest

from rag_blast.manifest import (
    ManifestLoadError,
    load_manifest,
    starter_manifest,
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
