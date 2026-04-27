from rag_blast.diff import diff_manifests


def test_diff_manifests_reports_nested_changes() -> None:
    old = {"embedding": {"model": "text-embedding-ada-002", "dimensions": 1536}}
    new = {"embedding": {"model": "text-embedding-3-large", "dimensions": 3072}}

    changes = diff_manifests(old, new)

    assert [change.path for change in changes] == [
        "embedding.dimensions",
        "embedding.model",
    ]


def test_diff_manifests_is_empty_for_equal_inputs() -> None:
    manifest = {"app": "customer-support-rag"}

    assert diff_manifests(manifest, manifest) == []


def test_diff_manifests_distinguishes_missing_from_literal_value() -> None:
    changes = diff_manifests({"app": "<missing>"}, {})

    assert len(changes) == 1
    assert changes[0].path == "app"
    assert changes[0].to_dict() == {"path": "app", "old": "<missing>", "new": "<missing key>"}
