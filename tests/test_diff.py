from copy import deepcopy

import pytest

from rag_blast.diff import ManifestDiff, diff_manifests
from rag_blast.manifest import starter_manifest


def test_diff_manifests_reports_nested_changes() -> None:
    old = {"embedding": {"model": "text-embedding-ada-002", "dimensions": 1536}}
    new = {"embedding": {"model": "text-embedding-3-large", "dimensions": 3072}}

    manifest_diff = diff_manifests(old, new)

    assert isinstance(manifest_diff, ManifestDiff)
    assert [change.path for change in manifest_diff.changes] == [
        "embedding.dimensions",
        "embedding.model",
    ]
    assert [change.category for change in manifest_diff.changes] == [
        "embedding_dimensions_changed",
        "embedding_model_changed",
    ]


def test_diff_manifests_is_empty_for_equal_inputs() -> None:
    manifest = {"app": "customer-support-rag"}

    manifest_diff = diff_manifests(manifest, manifest)

    assert manifest_diff.change_count == 0
    assert manifest_diff.changes == ()
    assert manifest_diff.categories == ()


def test_diff_manifests_distinguishes_missing_from_literal_value() -> None:
    manifest_diff = diff_manifests({"app": "<missing>"}, {})

    assert manifest_diff.change_count == 1
    assert manifest_diff.changes[0].path == "app"
    assert manifest_diff.changes[0].to_dict() == {
        "path": "app",
        "category": "manifest_field_changed",
        "summary": "Manifest field changed",
        "old": "<missing>",
        "new": "<missing key>",
    }


@pytest.mark.parametrize(
    ("path", "new_value", "category"),
    [
        ("embedding.provider", "voyage", "embedding_provider_changed"),
        ("embedding.model", "text-embedding-3-large", "embedding_model_changed"),
        ("embedding.dimensions", 3072, "embedding_dimensions_changed"),
        ("chunking.strategy", "markdown", "chunking_strategy_changed"),
        ("chunking.chunk_size", 1200, "chunk_size_changed"),
        ("chunking.chunk_overlap", 150, "chunk_overlap_changed"),
        ("vector_store.provider", "weaviate", "vector_store_provider_changed"),
        ("vector_store.collection", "SupportDocs_v4", "vector_collection_changed"),
        ("retriever.top_k", 12, "retriever_top_k_changed"),
        ("retriever.hybrid", True, "hybrid_retrieval_changed"),
    ],
)
def test_diff_manifests_categorizes_core_field_changes(
    path: str, new_value: object, category: str
) -> None:
    old = starter_manifest()
    new = deepcopy(old)
    _set_path(new, path, new_value)

    manifest_diff = diff_manifests(old, new)

    assert (path, category) in [(change.path, change.category) for change in manifest_diff.changes]
    if not path.startswith("embedding."):
        assert manifest_diff.change_count == 1


def test_diff_manifests_detects_reranker_added() -> None:
    old = starter_manifest()
    new = deepcopy(old)
    new["retriever"]["reranker"] = {"provider": "cohere", "model": "rerank-english-v3.0"}

    manifest_diff = diff_manifests(old, new)

    assert [(change.path, change.category) for change in manifest_diff.changes] == [
        ("retriever.reranker", "reranker_added")
    ]


def test_diff_manifests_detects_reranker_removed() -> None:
    old = starter_manifest()
    old["retriever"]["reranker"] = {"provider": "cohere", "model": "rerank-english-v3.0"}
    new = deepcopy(old)
    new["retriever"]["reranker"] = None

    manifest_diff = diff_manifests(old, new)

    assert [(change.path, change.category) for change in manifest_diff.changes] == [
        ("retriever.reranker", "reranker_removed")
    ]


def test_diff_manifests_treats_missing_and_null_reranker_as_equal() -> None:
    old = {"retriever": {"reranker": None}}
    new = {"retriever": {}}

    assert diff_manifests(old, new).change_count == 0
    assert diff_manifests(new, old).change_count == 0


def test_diff_manifests_classifies_reranker_removed_when_new_state_is_empty() -> None:
    old = {"retriever": {"reranker": {"model": "rerank-english-v3.0"}}}
    new = {"retriever": {}}

    manifest_diff = diff_manifests(old, new)

    assert [(change.path, change.category) for change in manifest_diff.changes] == [
        ("retriever.reranker", "reranker_removed")
    ]


def test_diff_manifests_detects_nested_reranker_changes() -> None:
    old = starter_manifest()
    old["retriever"]["reranker"] = {"provider": "cohere", "model": "rerank-english-v3.0"}
    new = deepcopy(old)
    new["retriever"]["reranker"]["model"] = "rerank-v3.5"

    manifest_diff = diff_manifests(old, new)

    assert [(change.path, change.category) for change in manifest_diff.changes] == [
        ("retriever.reranker.model", "reranker_changed")
    ]


def test_diff_manifests_detects_nested_reranker_field_added_as_changed() -> None:
    old = starter_manifest()
    old["retriever"]["reranker"] = {"model": "rerank-english-v3.0"}
    new = deepcopy(old)
    new["retriever"]["reranker"]["provider"] = "cohere"

    manifest_diff = diff_manifests(old, new)

    assert [(change.path, change.category) for change in manifest_diff.changes] == [
        ("retriever.reranker.provider", "reranker_changed")
    ]


def test_diff_manifests_detects_nested_reranker_field_removed_as_changed() -> None:
    old = starter_manifest()
    old["retriever"]["reranker"] = {"provider": "cohere", "model": "rerank-english-v3.0"}
    new = deepcopy(old)
    del new["retriever"]["reranker"]["provider"]

    manifest_diff = diff_manifests(old, new)

    assert [(change.path, change.category) for change in manifest_diff.changes] == [
        ("retriever.reranker.provider", "reranker_changed")
    ]


def test_diff_manifests_detects_eval_dataset_removed() -> None:
    old = starter_manifest()
    new = deepcopy(old)
    new["evals"] = []

    manifest_diff = diff_manifests(old, new)

    assert [(change.path, change.category) for change in manifest_diff.changes] == [
        ("evals[retrieval_golden]", "eval_dataset_missing")
    ]


def test_diff_manifests_detects_eval_dataset_path_changed() -> None:
    old = starter_manifest()
    new = deepcopy(old)
    new["evals"][0]["path"] = "evals/retrieval_golden_v2.jsonl"

    manifest_diff = diff_manifests(old, new)

    assert [(change.path, change.category) for change in manifest_diff.changes] == [
        ("evals[retrieval_golden].path", "eval_dataset_changed")
    ]


def test_diff_manifests_falls_back_for_duplicate_eval_names() -> None:
    old = starter_manifest()
    new = deepcopy(old)
    old["evals"].append({"name": "retrieval_golden", "path": "evals/duplicate.jsonl"})
    new["evals"][0]["path"] = "evals/retrieval_golden_v2.jsonl"

    manifest_diff = diff_manifests(old, new)

    assert [(change.path, change.category) for change in manifest_diff.changes] == [
        ("evals", "manifest_field_changed")
    ]


def test_diff_manifests_ignores_eval_order() -> None:
    old = starter_manifest()
    old["evals"].append({"name": "answer_quality", "path": "evals/answer_quality.jsonl"})
    new = deepcopy(old)
    new["evals"] = list(reversed(new["evals"]))

    manifest_diff = diff_manifests(old, new)

    assert manifest_diff.change_count == 0


def test_diff_manifests_ignores_cache_order() -> None:
    old = starter_manifest()
    old["caches"].append(
        {
            "type": "embedding_cache",
            "namespace": "support_embedding_cache_v1",
            "embedding_model": "text-embedding-ada-002",
        }
    )
    new = deepcopy(old)
    new["caches"] = list(reversed(new["caches"]))

    manifest_diff = diff_manifests(old, new)

    assert manifest_diff.change_count == 0


def test_diff_manifests_detects_semantic_cache_namespace_unchanged_after_embedding_change() -> None:
    old = starter_manifest()
    new = deepcopy(old)
    new["embedding"]["model"] = "text-embedding-3-large"
    new["caches"][0]["embedding_model"] = "text-embedding-3-large"

    manifest_diff = diff_manifests(old, new)

    assert ("caches[support_rag_prod_v4].namespace", "semantic_cache_namespace_unchanged") in [
        (change.path, change.category) for change in manifest_diff.changes
    ]


def _set_path(manifest: dict[str, object], path: str, value: object) -> None:
    current: dict[str, object] = manifest
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]  # type: ignore[assignment]
    current[parts[-1]] = value
