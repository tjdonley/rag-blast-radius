from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
_MISSING = object()


class IntegrationScanError(Exception):
    """Raised when an integration source cannot be scanned."""


@dataclass(frozen=True)
class DiscoveredValue:
    value: Any
    location: str


@dataclass(frozen=True)
class IntegrationScan:
    manifest: dict[str, Any]
    warnings: list[str]
    scanned_files: list[Path]


def scan_llamaindex_qdrant(source: Path) -> IntegrationScan:
    """Scan Python source for common LlamaIndex + Qdrant config patterns."""
    files = _python_files(source)
    values: dict[str, list[DiscoveredValue]] = {}
    warnings: list[str] = []

    for file_path in files:
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except SyntaxError as error:
            warnings.append(f"Skipped {file_path}: Python parse error on line {error.lineno}.")
            continue
        except OSError as error:
            warnings.append(f"Skipped {file_path}: {error}.")
            continue

        visitor = _LlamaIndexQdrantVisitor(file_path, _collect_constants(tree))
        visitor.visit(tree)
        warnings.extend(visitor.warnings)
        for field_path, discovered_values in visitor.values.items():
            values.setdefault(field_path, []).extend(discovered_values)

    manifest, manifest_warnings = _build_partial_manifest(values)
    warnings.extend(manifest_warnings)
    return IntegrationScan(manifest=manifest, warnings=warnings, scanned_files=files)


def render_partial_manifest(manifest: dict[str, Any]) -> str:
    """Render a partial manifest draft as stable JSON."""
    return json.dumps(manifest, indent=2) + "\n"


class _LlamaIndexQdrantVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path, constants: dict[str, Any]) -> None:
        self.file_path = file_path
        self.constants = constants
        self.values: dict[str, list[DiscoveredValue]] = {}
        self.warnings: list[str] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_settings_assignment(target, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record_settings_assignment(node.target, node.value, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        class_name = call_name.rsplit(".", 1)[-1]
        chunking_factory_strategy = _chunking_factory_strategy(call_name)

        if class_name == "QdrantVectorStore":
            self._record_qdrant_vector_store(node)
        elif class_name in {"OpenAIEmbedding", "AzureOpenAIEmbedding"}:
            self._record_openai_embedding(node, provider=_embedding_provider(class_name))
        elif class_name in {"HuggingFaceEmbedding", "SentenceTransformerEmbedding"}:
            self._record_named_embedding(node, provider=_embedding_provider(class_name))
        elif class_name in {"SentenceSplitter", "TokenTextSplitter", "SimpleNodeParser"}:
            self._record_chunking(node, strategy=_chunking_strategy(class_name))
        elif chunking_factory_strategy is not None:
            self._record_chunking(node, strategy=chunking_factory_strategy)
        elif class_name == "VectorIndexRetriever" or call_name.endswith(".as_retriever"):
            self._record_retriever(node)
        elif call_name.endswith("RetrieverQueryEngine.from_args"):
            self._record_retriever(node)
        elif class_name in {"SentenceTransformerRerank", "CohereRerank", "LLMRerank"}:
            self._record_reranker(node, provider=_reranker_provider(class_name))

        self.generic_visit(node)

    def _record_qdrant_vector_store(self, node: ast.Call) -> None:
        self._add_string("vector_store.provider", "qdrant", node)
        collection = self._keyword_value(node, "collection_name", "collection")
        self._add_string("vector_store.collection", collection, node)

        alias = self._keyword_value(node, "alias_name", "alias")
        self._add_string("vector_store.alias", alias, node)

        hybrid = self._keyword_value(node, "enable_hybrid", "hybrid")
        self._add_bool("retriever.hybrid", hybrid, node)

    def _record_openai_embedding(self, node: ast.Call, *, provider: str) -> None:
        self._add_string("embedding.provider", provider, node)

        model = self._keyword_value(node, "model", "model_name", "deployment_name", "engine")
        if model is _MISSING and node.args:
            model = _literal_value(node.args[0], self.constants)
        self._add_string("embedding.model", model, node)

        dimensions = self._keyword_value(node, "dimensions", "embed_dim")
        if dimensions is _MISSING and isinstance(model, str):
            dimensions = OPENAI_EMBEDDING_DIMENSIONS.get(model, _MISSING)
        self._add_positive_int("embedding.dimensions", dimensions, node)

    def _record_named_embedding(self, node: ast.Call, *, provider: str) -> None:
        self._add_string("embedding.provider", provider, node)
        model = self._keyword_value(node, "model_name", "model")
        if model is _MISSING and node.args:
            model = _literal_value(node.args[0], self.constants)
        self._add_string("embedding.model", model, node)

    def _record_chunking(self, node: ast.Call, *, strategy: str) -> None:
        self._add_string("chunking.strategy", strategy, node)
        chunk_size = self._keyword_value(node, "chunk_size")
        self._add_positive_int("chunking.chunk_size", chunk_size, node)
        chunk_overlap = self._keyword_value(node, "chunk_overlap")
        self._add_non_negative_int("chunking.chunk_overlap", chunk_overlap, node)

    def _record_retriever(self, node: ast.Call) -> None:
        top_k = self._keyword_value(node, "similarity_top_k", "top_k", "k")
        self._add_positive_int("retriever.top_k", top_k, node)

        query_mode = self._keyword_value(node, "vector_store_query_mode", "query_mode")
        if isinstance(query_mode, str):
            self._add_bool("retriever.hybrid", query_mode.lower() == "hybrid", node)

        hybrid = self._keyword_value(node, "hybrid", "enable_hybrid")
        self._add_bool("retriever.hybrid", hybrid, node)

    def _record_reranker(self, node: ast.Call, *, provider: str | None) -> None:
        if provider is not None:
            self._add_string("retriever.reranker.provider", provider, node)
        model = self._keyword_value(node, "model", "model_name")
        self._add_string("retriever.reranker.model", model, node)

    def _record_settings_assignment(
        self, target: ast.expr, value_node: ast.expr, location_node: ast.AST
    ) -> None:
        target_name = _attribute_name(target)
        value = _literal_value(value_node, self.constants)
        if target_name.endswith("Settings.chunk_size"):
            self._add_positive_int("chunking.chunk_size", value, location_node)
            self._add_string("chunking.strategy", "llamaindex_settings", location_node)
        elif target_name.endswith("Settings.chunk_overlap"):
            self._add_non_negative_int("chunking.chunk_overlap", value, location_node)
            self._add_string("chunking.strategy", "llamaindex_settings", location_node)

    def _keyword_value(self, node: ast.Call, *names: str) -> Any:
        for keyword in node.keywords:
            if keyword.arg in names:
                return _literal_value(keyword.value, self.constants)
        return _MISSING

    def _add_string(self, field_path: str, value: Any, node: ast.AST) -> None:
        if value is _MISSING:
            return
        if isinstance(value, str) and value.strip():
            self._add_value(field_path, value, node)
            return
        self.warnings.append(
            f"Ignored {field_path} at {self._location(node)}: expected a string literal."
        )

    def _add_positive_int(self, field_path: str, value: Any, node: ast.AST) -> None:
        if value is _MISSING:
            return
        if type(value) is int and value > 0:
            self._add_value(field_path, value, node)
            return
        self.warnings.append(
            f"Ignored {field_path} at {self._location(node)}: expected a positive integer literal."
        )

    def _add_non_negative_int(self, field_path: str, value: Any, node: ast.AST) -> None:
        if value is _MISSING:
            return
        if type(value) is int and value >= 0:
            self._add_value(field_path, value, node)
            return
        self.warnings.append(
            f"Ignored {field_path} at {self._location(node)}: expected a non-negative integer literal."
        )

    def _add_bool(self, field_path: str, value: Any, node: ast.AST) -> None:
        if value is _MISSING:
            return
        if type(value) is bool:
            self._add_value(field_path, value, node)
            return
        self.warnings.append(
            f"Ignored {field_path} at {self._location(node)}: expected a bool literal."
        )

    def _add_value(self, field_path: str, value: Any, node: ast.AST) -> None:
        self.values.setdefault(field_path, []).append(DiscoveredValue(value, self._location(node)))

    def _location(self, node: ast.AST) -> str:
        return f"{self.file_path}:{getattr(node, 'lineno', '?')}"


def _build_partial_manifest(
    values: dict[str, list[DiscoveredValue]],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    manifest: dict[str, Any] = {}

    for field_path in (
        "embedding.provider",
        "embedding.model",
        "embedding.dimensions",
        "chunking.strategy",
        "chunking.chunk_size",
        "chunking.chunk_overlap",
        "vector_store.provider",
        "vector_store.collection",
        "vector_store.alias",
        "retriever.top_k",
        "retriever.hybrid",
        "retriever.reranker.provider",
        "retriever.reranker.model",
    ):
        value = _select_value(field_path, values, warnings)
        if value is not _MISSING:
            _set_nested_value(manifest, field_path, value)

    if "retriever.reranker.model" in values and "retriever.reranker.provider" not in values:
        _set_nested_value(manifest, "retriever.reranker.provider", None)

    if _has_nested_value(manifest, "retriever.reranker.provider") and not _has_nested_value(
        manifest, "retriever.reranker.model"
    ):
        warnings.append(
            "retriever.reranker.model is required when a reranker is detected and must be filled "
            "manually."
        )

    for required_field in (
        "app",
        "environment",
        "embedding.provider",
        "embedding.model",
        "embedding.dimensions",
        "chunking.strategy",
        "chunking.chunk_size",
        "chunking.chunk_overlap",
        "vector_store.provider",
        "vector_store.collection",
        "retriever.top_k",
        "retriever.hybrid",
    ):
        if not _has_nested_value(manifest, required_field):
            warnings.append(f"{required_field} is required and must be filled manually.")

    warnings.append("caches are not inferred; add semantic cache entries manually if used.")
    warnings.append("evals are not inferred; add retrieval eval paths manually if available.")
    return manifest, warnings


def _select_value(
    field_path: str,
    values: dict[str, list[DiscoveredValue]],
    warnings: list[str],
) -> Any:
    discovered_values = values.get(field_path, [])
    if not discovered_values:
        return _MISSING

    first = discovered_values[0]
    unique_values = {json.dumps(value.value, sort_keys=True) for value in discovered_values}
    if len(unique_values) > 1:
        warnings.append(
            f"Multiple values found for {field_path}; using {first.value!r} from {first.location}."
        )
    return first.value


def _set_nested_value(manifest: dict[str, Any], field_path: str, value: Any) -> None:
    current = manifest
    parts = field_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _has_nested_value(manifest: dict[str, Any], field_path: str) -> bool:
    current: Any = manifest
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _python_files(source: Path) -> list[Path]:
    if not source.exists():
        raise IntegrationScanError(f"Source does not exist: {source}")

    if source.is_file():
        if source.suffix != ".py":
            raise IntegrationScanError(f"Source must be a Python file or directory: {source}")
        return [source]

    files = [path for path in sorted(source.rglob("*.py")) if not _is_skipped(path, source)]
    if not files:
        raise IntegrationScanError(f"No Python files found under source: {source}")
    return files


def _is_skipped(path: Path, source: Path) -> bool:
    relative_parts = path.relative_to(source).parts[:-1]
    return any(part in SKIP_DIR_NAMES or part.startswith(".") for part in relative_parts)


def _collect_constants(tree: ast.AST) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _literal_value(node.value, constants)
            if value is _MISSING:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _literal_value(node.value, constants) if node.value is not None else _MISSING
            if value is not _MISSING:
                constants[node.target.id] = value
    return constants


def _literal_value(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) in {str, int, bool}:
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id, _MISSING)
    if isinstance(node, ast.Call) and _call_name(node.func) in {"os.getenv", "getenv"}:
        if len(node.args) >= 2:
            return _literal_value(node.args[1], constants)
    return _MISSING


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _embedding_provider(class_name: str) -> str:
    providers = {
        "AzureOpenAIEmbedding": "azure_openai",
        "HuggingFaceEmbedding": "huggingface",
        "OpenAIEmbedding": "openai",
        "SentenceTransformerEmbedding": "sentence_transformers",
    }
    return providers[class_name]


def _chunking_strategy(class_name: str) -> str:
    strategies = {
        "SentenceSplitter": "sentence_splitter",
        "SimpleNodeParser": "simple_node_parser",
        "TokenTextSplitter": "token_text_splitter",
    }
    return strategies[class_name]


def _chunking_factory_strategy(call_name: str) -> str | None:
    for class_name in ("SentenceSplitter", "SimpleNodeParser", "TokenTextSplitter"):
        if call_name.endswith(f"{class_name}.from_defaults"):
            return _chunking_strategy(class_name)
    return None


def _reranker_provider(class_name: str) -> str | None:
    providers = {
        "CohereRerank": "cohere",
        "SentenceTransformerRerank": "sentence_transformers",
        "LLMRerank": None,
    }
    return providers[class_name]
