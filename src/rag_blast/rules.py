from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_blast.diff import ManifestDiff


@dataclass(frozen=True)
class RuleInfo:
    """Human-readable metadata for a blast-radius rule."""

    id: str
    severity: str
    summary: str
    recommendation: str


@dataclass(frozen=True)
class RuleFinding:
    """A triggered blast-radius rule with the changes that caused it."""

    rule_id: str
    severity: str
    summary: str
    recommendation: str
    change_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "change_paths": list(self.change_paths),
        }


RULES: dict[str, RuleInfo] = {
    "REEMBED_REQUIRED": RuleInfo(
        id="REEMBED_REQUIRED",
        severity="HIGH",
        summary="Embedding provider, model, dimensions, or chunking changed.",
        recommendation=(
            "Rebuild document embeddings before serving traffic from the changed "
            "embedding or chunking configuration."
        ),
    ),
    "VECTOR_INDEX_INCOMPATIBLE": RuleInfo(
        id="VECTOR_INDEX_INCOMPATIBLE",
        severity="HIGH",
        summary="Existing vectors may not be comparable to new query vectors.",
        recommendation="Build a shadow index or preserve the old index until migration is complete.",
    ),
    "SEMANTIC_CACHE_UNSAFE": RuleInfo(
        id="SEMANTIC_CACHE_UNSAFE",
        severity="HIGH",
        summary="Semantic cache entries may have been produced with stale embeddings or retrieval behavior.",
        recommendation="Use a new cache namespace when embedding or retrieval behavior changes.",
    ),
    "RETRIEVAL_BASELINE_STALE": RuleInfo(
        id="RETRIEVAL_BASELINE_STALE",
        severity="MEDIUM",
        summary="Retrieval eval baselines may no longer describe the proposed system.",
        recommendation="Replay representative retrieval evals before rollout.",
    ),
    "CHUNKING_CHANGED": RuleInfo(
        id="CHUNKING_CHANGED",
        severity="HIGH",
        summary="Chunking strategy, size, or overlap changed.",
        recommendation="Regenerate chunks, rebuild document embeddings, and replay retrieval evals.",
    ),
    "RERANKER_CHANGED": RuleInfo(
        id="RERANKER_CHANGED",
        severity="MEDIUM",
        summary="Reranker behavior changed.",
        recommendation="Replay retrieval and answer-quality evals before rollout.",
    ),
    "RETRIEVER_BEHAVIOR_CHANGED": RuleInfo(
        id="RETRIEVER_BEHAVIOR_CHANGED",
        severity="MEDIUM",
        summary="Retriever behavior changed.",
        recommendation="Compare retrieval overlap and answer quality against the current baseline.",
    ),
    "SHADOW_INDEX_RECOMMENDED": RuleInfo(
        id="SHADOW_INDEX_RECOMMENDED",
        severity="MEDIUM",
        summary="The change should be rolled out through a shadow index.",
        recommendation="Build the proposed index side-by-side and replay representative queries.",
    ),
    "ROLLBACK_REQUIRES_OLD_INDEX": RuleInfo(
        id="ROLLBACK_REQUIRES_OLD_INDEX",
        severity="MEDIUM",
        summary="Rollback depends on preserving the old index.",
        recommendation="Keep the previous index and cache namespace until the rollback window closes.",
    ),
}

RULE_ORDER = tuple(RULES)
SEVERITY_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

EMBEDDING_CATEGORIES = frozenset(
    {
        "embedding_provider_changed",
        "embedding_model_changed",
        "embedding_dimensions_changed",
    }
)
CHUNKING_CATEGORIES = frozenset(
    {
        "chunking_strategy_changed",
        "chunk_size_changed",
        "chunk_overlap_changed",
    }
)
VECTOR_INDEX_CATEGORIES = frozenset(
    {
        "vector_store_provider_changed",
        "vector_collection_changed",
    }
)
RETRIEVER_CATEGORIES = frozenset(
    {
        "retriever_top_k_changed",
        "hybrid_retrieval_changed",
    }
)
RERANKER_CATEGORIES = frozenset(
    {
        "reranker_added",
        "reranker_removed",
        "reranker_changed",
    }
)
EVAL_CATEGORIES = frozenset(
    {
        "eval_dataset_missing",
        "eval_dataset_changed",
    }
)
SEMANTIC_CACHE_CATEGORIES = frozenset(
    {
        "semantic_cache_namespace_unchanged",
    }
)

RULE_TRIGGERS: dict[str, frozenset[str]] = {
    "REEMBED_REQUIRED": EMBEDDING_CATEGORIES | CHUNKING_CATEGORIES,
    "VECTOR_INDEX_INCOMPATIBLE": EMBEDDING_CATEGORIES,
    "SEMANTIC_CACHE_UNSAFE": SEMANTIC_CACHE_CATEGORIES,
    "RETRIEVAL_BASELINE_STALE": EMBEDDING_CATEGORIES
    | CHUNKING_CATEGORIES
    | RETRIEVER_CATEGORIES
    | RERANKER_CATEGORIES
    | EVAL_CATEGORIES,
    "CHUNKING_CHANGED": CHUNKING_CATEGORIES,
    "RERANKER_CHANGED": RERANKER_CATEGORIES,
    "RETRIEVER_BEHAVIOR_CHANGED": RETRIEVER_CATEGORIES | RERANKER_CATEGORIES,
    "SHADOW_INDEX_RECOMMENDED": EMBEDDING_CATEGORIES
    | CHUNKING_CATEGORIES
    | VECTOR_INDEX_CATEGORIES,
    "ROLLBACK_REQUIRES_OLD_INDEX": EMBEDDING_CATEGORIES
    | CHUNKING_CATEGORIES
    | VECTOR_INDEX_CATEGORIES,
}


def get_rule(rule_id: str) -> RuleInfo | None:
    """Return rule metadata by identifier."""
    return RULES.get(rule_id.upper())


def evaluate_rules(manifest_diff: ManifestDiff) -> tuple[RuleFinding, ...]:
    """Return deterministic rule findings for a manifest diff."""
    findings: list[RuleFinding] = []
    for rule_id in RULE_ORDER:
        matching_paths = _matching_paths(manifest_diff, RULE_TRIGGERS[rule_id])
        if not matching_paths:
            continue

        rule = RULES[rule_id]
        findings.append(
            RuleFinding(
                rule_id=rule.id,
                severity=rule.severity,
                summary=rule.summary,
                recommendation=rule.recommendation,
                change_paths=matching_paths,
            )
        )

    return tuple(findings)


def highest_severity(findings: tuple[RuleFinding, ...]) -> str:
    """Return the highest severity across triggered findings."""
    if not findings:
        return "NONE"

    return max(
        (finding.severity for finding in findings), key=lambda severity: SEVERITY_ORDER[severity]
    )


def _matching_paths(manifest_diff: ManifestDiff, categories: frozenset[str]) -> tuple[str, ...]:
    return tuple(change.path for change in manifest_diff.changes if change.category in categories)
