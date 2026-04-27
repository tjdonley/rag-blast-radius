from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class _MissingValue:
    pass


MISSING = _MissingValue()


@dataclass(frozen=True)
class ManifestChange:
    """A single changed path between two manifests."""

    path: str
    old: Any
    new: Any

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "old": _display_value(self.old), "new": _display_value(self.new)}


def diff_manifests(old: dict[str, Any], new: dict[str, Any]) -> list[ManifestChange]:
    """Return a deterministic list of changes between two manifest dictionaries."""
    return list(_diff_values(old, new))


def _diff_values(old: Any, new: Any, path: str = "") -> list[ManifestChange]:
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[ManifestChange] = []
        for key in sorted(set(old) | set(new)):
            child_path = f"{path}.{key}" if path else str(key)
            changes.extend(_diff_values(old.get(key, MISSING), new.get(key, MISSING), child_path))
        return changes

    if old == new:
        return []

    return [ManifestChange(path=path or "<root>", old=old, new=new)]


def _display_value(value: Any) -> Any:
    if value is MISSING:
        return "<missing key>"
    return value
