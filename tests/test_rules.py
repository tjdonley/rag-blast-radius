from copy import deepcopy

import pytest

from rag_blast.diff import diff_manifests
from rag_blast.manifest import starter_manifest
from rag_blast.rules import RULES, evaluate_rules, get_rule, highest_severity

EXPECTED_RULE_IDS = {
    "REEMBED_REQUIRED",
    "VECTOR_INDEX_INCOMPATIBLE",
    "SEMANTIC_CACHE_UNSAFE",
    "RETRIEVAL_BASELINE_STALE",
    "CHUNKING_CHANGED",
    "RERANKER_CHANGED",
    "RETRIEVER_BEHAVIOR_CHANGED",
    "SHADOW_INDEX_RECOMMENDED",
    "ROLLBACK_REQUIRES_OLD_INDEX",
}


def test_get_rule_is_case_insensitive() -> None:
    rule = get_rule("reembed_required")

    assert rule is not None
    assert rule.id == "REEMBED_REQUIRED"


def test_get_rule_returns_none_for_unknown_rule() -> None:
    assert get_rule("UNKNOWN_RULE") is None


def test_all_phase_4_rules_have_understandable_metadata() -> None:
    assert set(RULES) == EXPECTED_RULE_IDS

    for rule in RULES.values():
        assert rule.id
        assert rule.severity in {"LOW", "MEDIUM", "HIGH"}
        assert rule.summary.endswith(".")
        assert rule.recommendation.endswith(".")


@pytest.mark.parametrize(
    ("rule_id", "path", "new_value"),
    [
        ("REEMBED_REQUIRED", "embedding.model", "text-embedding-3-large"),
        ("VECTOR_INDEX_INCOMPATIBLE", "embedding.model", "text-embedding-3-large"),
        ("SEMANTIC_CACHE_UNSAFE", "embedding.model", "text-embedding-3-large"),
        ("RETRIEVAL_BASELINE_STALE", "evals[0].path", "evals/retrieval_golden_v2.jsonl"),
        ("CHUNKING_CHANGED", "chunking.chunk_size", 1200),
        ("RERANKER_CHANGED", "retriever.reranker", {"model": "rerank-english-v3.0"}),
        ("RETRIEVER_BEHAVIOR_CHANGED", "retriever.top_k", 12),
        ("SHADOW_INDEX_RECOMMENDED", "vector_store.collection", "support_docs_v4"),
        ("ROLLBACK_REQUIRES_OLD_INDEX", "vector_store.collection", "support_docs_v4"),
    ],
)
def test_evaluate_rules_triggers_each_phase_4_rule(
    rule_id: str, path: str, new_value: object
) -> None:
    manifest_diff = _diff_with_change(path, new_value)

    assert rule_id in _finding_ids(manifest_diff)


def test_evaluate_rules_returns_deterministic_rule_order() -> None:
    manifest_diff = _diff_with_change("embedding.model", "text-embedding-3-large")

    assert _finding_ids(manifest_diff) == [
        "REEMBED_REQUIRED",
        "VECTOR_INDEX_INCOMPATIBLE",
        "SEMANTIC_CACHE_UNSAFE",
        "RETRIEVAL_BASELINE_STALE",
        "SHADOW_INDEX_RECOMMENDED",
        "ROLLBACK_REQUIRES_OLD_INDEX",
    ]


@pytest.mark.parametrize(
    ("path", "new_value", "expected_rule_ids"),
    [
        (
            "vector_store.collection",
            "support_docs_v4",
            ["SHADOW_INDEX_RECOMMENDED", "ROLLBACK_REQUIRES_OLD_INDEX"],
        ),
        (
            "retriever.top_k",
            12,
            [
                "SEMANTIC_CACHE_UNSAFE",
                "RETRIEVAL_BASELINE_STALE",
                "RETRIEVER_BEHAVIOR_CHANGED",
            ],
        ),
        ("evals[0].path", "evals/retrieval_golden_v2.jsonl", ["RETRIEVAL_BASELINE_STALE"]),
        (
            "chunking.chunk_size",
            1200,
            [
                "REEMBED_REQUIRED",
                "RETRIEVAL_BASELINE_STALE",
                "CHUNKING_CHANGED",
                "SHADOW_INDEX_RECOMMENDED",
                "ROLLBACK_REQUIRES_OLD_INDEX",
            ],
        ),
    ],
)
def test_evaluate_rules_returns_expected_rule_sets(
    path: str, new_value: object, expected_rule_ids: list[str]
) -> None:
    assert _finding_ids(_diff_with_change(path, new_value)) == expected_rule_ids


def test_evaluate_rules_includes_triggering_change_paths() -> None:
    findings = evaluate_rules(_diff_with_change("chunking.chunk_size", 1200))
    chunking_finding = next(
        finding for finding in findings if finding.rule_id == "CHUNKING_CHANGED"
    )

    assert chunking_finding.change_paths == ("chunking.chunk_size",)


def test_highest_severity_returns_none_without_findings() -> None:
    assert highest_severity(()) == "NONE"


def test_highest_severity_returns_highest_triggered_severity() -> None:
    findings = evaluate_rules(_diff_with_change("embedding.model", "text-embedding-3-large"))

    assert highest_severity(findings) == "HIGH"


def _diff_with_change(path: str, value: object):
    old = starter_manifest()
    new = deepcopy(old)
    _set_path(new, path, value)
    return diff_manifests(old, new)


def _finding_ids(manifest_diff) -> list[str]:
    return [finding.rule_id for finding in evaluate_rules(manifest_diff)]


def _set_path(manifest: dict[str, object], path: str, value: object) -> None:
    current: object = manifest
    parts = path.split(".")
    for part in parts[:-1]:
        if "[" in part:
            name, index = part.rstrip("]").split("[")
            current = current[name][int(index)]  # type: ignore[index]
        else:
            current = current[part]  # type: ignore[index]

    leaf = parts[-1]
    if "[" in leaf:
        name, index = leaf.rstrip("]").split("[")
        current[name][int(index)] = value  # type: ignore[index]
    else:
        current[leaf] = value  # type: ignore[index]
