from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class _MissingValue:
    pass


MISSING = _MissingValue()


@dataclass(frozen=True)
class ManifestChange:
    """A single semantic change between two manifests."""

    path: str
    old: Any
    new: Any
    category: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "category": self.category,
            "summary": self.summary,
            "old": _display_value(self.old),
            "new": _display_value(self.new),
        }


@dataclass(frozen=True)
class ManifestDiff:
    """Structured diff between two RAG manifests."""

    changes: tuple[ManifestChange, ...]

    @property
    def change_count(self) -> int:
        return len(self.changes)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({change.category for change in self.changes}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_count": self.change_count,
            "categories": list(self.categories),
            "changes": [change.to_dict() for change in self.changes],
        }


FIELD_CATEGORIES: dict[str, tuple[str, str]] = {
    "embedding.provider": ("embedding_provider_changed", "Embedding provider changed"),
    "embedding.model": ("embedding_model_changed", "Embedding model changed"),
    "embedding.dimensions": ("embedding_dimensions_changed", "Embedding dimensions changed"),
    "chunking.strategy": ("chunking_strategy_changed", "Chunking strategy changed"),
    "chunking.chunk_size": ("chunk_size_changed", "Chunk size changed"),
    "chunking.chunk_overlap": ("chunk_overlap_changed", "Chunk overlap changed"),
    "vector_store.provider": ("vector_store_provider_changed", "Vector store provider changed"),
    "vector_store.collection": ("vector_collection_changed", "Vector collection changed"),
    "retriever.top_k": ("retriever_top_k_changed", "Retriever top_k changed"),
    "retriever.hybrid": ("hybrid_retrieval_changed", "Hybrid retrieval setting changed"),
}

EMBEDDING_PATHS = frozenset({"embedding.provider", "embedding.model", "embedding.dimensions"})


def diff_manifests(old: dict[str, Any], new: dict[str, Any]) -> ManifestDiff:
    """Return a deterministic structured diff between two manifest dictionaries."""
    field_changes = list(_diff_values(old, new))
    derived_changes = list(_semantic_cache_namespace_changes(old, new, field_changes))
    changes = sorted([*field_changes, *derived_changes], key=lambda change: change.path)
    return ManifestDiff(changes=tuple(changes))


def _diff_values(old: Any, new: Any, path: str = "") -> list[ManifestChange]:
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[ManifestChange] = []
        for key in sorted(set(old) | set(new)):
            child_path = f"{path}.{key}" if path else str(key)
            changes.extend(_diff_values(old.get(key, MISSING), new.get(key, MISSING), child_path))
        return changes

    if isinstance(old, list) and isinstance(new, list):
        key_name = _list_identity_key(path, old, new)
        if key_name is not None:
            return _diff_keyed_lists(old, new, path, key_name)

    if old == new:
        return []

    return [_build_change(path=path or "<root>", old=old, new=new)]


def _build_change(path: str, old: Any, new: Any) -> ManifestChange:
    category, summary = _categorize_change(path, old, new)
    return ManifestChange(path=path, old=old, new=new, category=category, summary=summary)


def _categorize_change(path: str, old: Any, new: Any) -> tuple[str, str]:
    if path in FIELD_CATEGORIES:
        return FIELD_CATEGORIES[path]

    if path.startswith("retriever.reranker"):
        if path == "retriever.reranker":
            if old is MISSING or old is None:
                return "reranker_added", "Reranker added"
            if new is MISSING or new is None:
                return "reranker_removed", "Reranker removed"
        return "reranker_changed", "Reranker changed"

    if path.startswith("evals["):
        if new is MISSING:
            return "eval_dataset_missing", "Eval dataset removed"
        return "eval_dataset_changed", "Eval dataset changed"

    if path.startswith("caches["):
        return "cache_changed", "Cache configuration changed"

    return "manifest_field_changed", "Manifest field changed"


def _list_identity_key(path: str, old: list[Any], new: list[Any]) -> str | None:
    if (
        path == "evals"
        and _all_dicts_have_unique_key(old, "name")
        and _all_dicts_have_unique_key(new, "name")
    ):
        return "name"

    if (
        path == "caches"
        and _all_dicts_have_unique_key(old, "namespace")
        and _all_dicts_have_unique_key(new, "namespace")
    ):
        return "namespace"

    return None


def _all_dicts_have_unique_key(values: Iterable[Any], key: str) -> bool:
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict) or key not in value:
            return False

        identity = str(value[key])
        if identity in seen:
            return False
        seen.add(identity)

    return True


def _diff_keyed_lists(
    old: list[dict[str, Any]], new: list[dict[str, Any]], path: str, key_name: str
) -> list[ManifestChange]:
    old_by_key = {str(item[key_name]): item for item in old}
    new_by_key = {str(item[key_name]): item for item in new}
    changes: list[ManifestChange] = []

    for key in sorted(set(old_by_key) | set(new_by_key)):
        child_path = f"{path}[{key}]"
        changes.extend(
            _diff_values(old_by_key.get(key, MISSING), new_by_key.get(key, MISSING), child_path)
        )

    return changes


def _semantic_cache_namespace_changes(
    old: dict[str, Any], new: dict[str, Any], field_changes: list[ManifestChange]
) -> list[ManifestChange]:
    if not any(change.path in EMBEDDING_PATHS for change in field_changes):
        return []

    old_namespaces = _semantic_cache_namespaces(old)
    new_namespaces = _semantic_cache_namespaces(new)
    unchanged_namespaces = sorted(old_namespaces & new_namespaces)

    return [
        ManifestChange(
            path=f"caches[{namespace}].namespace",
            old=namespace,
            new=namespace,
            category="semantic_cache_namespace_unchanged",
            summary="Semantic cache namespace unchanged after embedding change",
        )
        for namespace in unchanged_namespaces
    ]


def _semantic_cache_namespaces(manifest: dict[str, Any]) -> set[str]:
    caches = manifest.get("caches", [])
    if not isinstance(caches, list):
        return set()

    return {
        str(cache["namespace"])
        for cache in caches
        if isinstance(cache, dict)
        and cache.get("type") == "semantic_cache"
        and isinstance(cache.get("namespace"), str)
    }


def _display_value(value: Any) -> Any:
    if value is MISSING:
        return "<missing key>"
    return value
