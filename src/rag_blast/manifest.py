from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator
from pydantic.types import NonNegativeInt, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ManifestLoadError(Exception):
    """Raised when a manifest cannot be loaded as a JSON object."""


class StrictModel(BaseModel):
    """Base model that rejects unknown manifest keys."""

    model_config = ConfigDict(extra="forbid", strict=True)


class EmbeddingConfig(StrictModel):
    provider: NonEmptyString
    model: NonEmptyString
    dimensions: PositiveInt


class ChunkingConfig(StrictModel):
    strategy: NonEmptyString
    chunk_size: PositiveInt
    chunk_overlap: NonNegativeInt

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class VectorStoreConfig(StrictModel):
    provider: NonEmptyString
    collection: NonEmptyString
    alias: NonEmptyString | None = None


class RerankerConfig(StrictModel):
    provider: NonEmptyString | None = None
    model: NonEmptyString


class RetrieverConfig(StrictModel):
    top_k: PositiveInt
    hybrid: bool
    reranker: RerankerConfig | None = None


class CacheConfig(StrictModel):
    type: NonEmptyString
    namespace: NonEmptyString
    embedding_model: NonEmptyString | None = None


class EvalConfig(StrictModel):
    name: NonEmptyString
    path: NonEmptyString


class RagManifest(StrictModel):
    app: NonEmptyString
    environment: NonEmptyString
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    vector_store: VectorStoreConfig
    retriever: RetrieverConfig
    caches: list[CacheConfig] = Field(default_factory=list)
    evals: list[EvalConfig] = Field(default_factory=list)


def starter_manifest() -> dict[str, Any]:
    """Return a starter RAG manifest users can edit."""
    manifest = {
        "app": "customer-support-rag",
        "environment": "prod",
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-ada-002",
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
            "reranker": None,
        },
        "caches": [
            {
                "type": "semantic_cache",
                "namespace": "support_rag_prod_v4",
                "embedding_model": "text-embedding-ada-002",
            }
        ],
        "evals": [
            {
                "name": "retrieval_golden",
                "path": "evals/retrieval_golden.jsonl",
            }
        ],
    }
    return validate_manifest(manifest)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a manifest from disk and ensure it is a JSON object."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestLoadError(f"Unable to read manifest {path}: {error}") from error

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ManifestLoadError(f"Invalid JSON in manifest {path}: {error.msg}") from error

    return validate_manifest(data, path=path)


def validate_manifest(data: Any, *, path: Path | None = None) -> dict[str, Any]:
    """Validate and normalize a manifest dictionary."""
    if not isinstance(data, dict):
        location = f": {path}" if path is not None else ""
        raise ManifestLoadError(f"Manifest must be a JSON object{location}")

    try:
        manifest = RagManifest.model_validate(data)
    except ValidationError as error:
        raise ManifestLoadError(_format_validation_errors(error, path=path)) from error

    return manifest.model_dump(mode="json")


def _format_validation_errors(error: ValidationError, *, path: Path | None = None) -> str:
    location = f" in manifest {path}" if path is not None else ""
    lines = [f"Validation failed{location}:"]
    for detail in error.errors():
        field_path = _format_error_location(detail["loc"])
        message = detail["msg"]
        lines.append(f"- {field_path}: {message}")
    return "\n".join(lines)


def _format_error_location(location: tuple[str | int, ...]) -> str:
    if not location:
        return "<root>"

    output = ""
    for part in location:
        if isinstance(part, int):
            output += f"[{part}]"
            continue

        if output:
            output += "."
        output += str(part)

    return output


def write_starter_manifest(path: Path, *, force: bool = False) -> None:
    """Write a starter manifest to disk."""
    if path.exists() and not force:
        raise FileExistsError(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(starter_manifest(), indent=2) + "\n", encoding="utf-8")
