from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleInfo:
    """Human-readable metadata for a blast-radius rule."""

    id: str
    severity: str
    summary: str
    recommendation: str


RULES: dict[str, RuleInfo] = {
    "REEMBED_REQUIRED": RuleInfo(
        id="REEMBED_REQUIRED",
        severity="HIGH",
        summary="Embedding provider, model, or dimensions changed.",
        recommendation="Rebuild document embeddings before serving query embeddings from the new model.",
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
}


def get_rule(rule_id: str) -> RuleInfo | None:
    """Return rule metadata by identifier."""
    return RULES.get(rule_id.upper())
