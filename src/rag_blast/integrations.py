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
SUPPORTED_CLASS_SUFFIXES = {
    "AzureOpenAIEmbedding": (".embeddings.azure_openai.AzureOpenAIEmbedding",),
    "CohereRerank": (".postprocessor.cohere_rerank.CohereRerank",),
    "HuggingFaceEmbedding": (".embeddings.huggingface.HuggingFaceEmbedding",),
    "LLMRerank": (".core.postprocessor.LLMRerank",),
    "OpenAIEmbedding": (".embeddings.openai.OpenAIEmbedding",),
    "QdrantVectorStore": (".vector_stores.qdrant.QdrantVectorStore",),
    "SentenceSplitter": (".core.node_parser.SentenceSplitter",),
    "SentenceTransformerEmbedding": (
        ".embeddings.huggingface.SentenceTransformerEmbedding",
        ".embeddings.sentence_transformer.SentenceTransformerEmbedding",
    ),
    "SentenceTransformerRerank": (".postprocessor.sbert_rerank.SentenceTransformerRerank",),
    "SimpleNodeParser": (".core.node_parser.SimpleNodeParser",),
    "TokenTextSplitter": (".core.node_parser.TokenTextSplitter",),
    "VectorIndexRetriever": (
        ".core.indices.vector_store.retrievers.VectorIndexRetriever",
        ".core.retrievers.VectorIndexRetriever",
    ),
}
SUPPORTED_INDEX_SUFFIXES = (
    ".core.VectorStoreIndex",
    ".core.indices.vector_store.base.VectorStoreIndex",
)
SUPPORTED_INDEX_FACTORY_METHODS = ("from_documents", "from_vector_store")
SUPPORTED_RETRIEVER_QUERY_ENGINE_SUFFIXES = (
    ".core.query_engine.RetrieverQueryEngine",
    ".core.query_engine.retriever_query_engine.RetrieverQueryEngine",
)
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
    scanned_files: list[Path] = []
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

        scanned_files.append(file_path)
        visitor = _LlamaIndexQdrantVisitor(file_path)
        visitor.visit(tree)
        warnings.extend(visitor.warnings)
        for field_path, discovered_values in visitor.values.items():
            values.setdefault(field_path, []).extend(discovered_values)

    if not values:
        warnings.append("No supported LlamaIndex + Qdrant configuration patterns were detected.")

    manifest, manifest_warnings = _build_partial_manifest(values)
    warnings.extend(manifest_warnings)
    return IntegrationScan(manifest=manifest, warnings=warnings, scanned_files=scanned_files)


def render_partial_manifest(manifest: dict[str, Any]) -> str:
    """Render a partial manifest draft as stable JSON."""
    return json.dumps(manifest, indent=2) + "\n"


class _LlamaIndexQdrantVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.scopes: list[dict[str, Any]] = [{}]
        self.import_alias_scopes: list[dict[str, str]] = [{}]
        self.index_name_scopes: list[set[str]] = [set()]
        self.predeclared_local_scopes: list[set[str]] = [set()]
        self.scope_kinds = ["module"]
        self.values: dict[str, list[DiscoveredValue]] = {}
        self.warnings: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".", 1)[0]
            if not alias.name.startswith("llama_index"):
                self._record_name_binding(local_name)
                continue

            qualified_name = alias.name if alias.asname else alias.name.split(".", 1)[0]
            self._record_import_alias(local_name, qualified_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if any(alias.name == "*" for alias in node.names):
            self._record_star_import_binding()
            return

        if node.level > 0 or not module.startswith("llama_index"):
            for alias in node.names:
                self._record_name_binding(alias.asname or alias.name)
            return

        for alias in node.names:
            self._record_import_alias(alias.asname or alias.name, f"{module}.{alias.name}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.import_alias_scopes[-1].pop(node.name, None)
        self.scopes[-1][node.name] = _MISSING
        self._visit_nested_scope(node, kind="function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.import_alias_scopes[-1].pop(node.name, None)
        self.scopes[-1][node.name] = _MISSING
        self._visit_nested_scope(node, kind="function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.import_alias_scopes[-1].pop(node.name, None)
        self.scopes[-1][node.name] = _MISSING
        self._visit_nested_scope(node, kind="class")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, (node.key, node.value))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.scopes.append({})
        self.import_alias_scopes.append({})
        self.index_name_scopes.append(set())
        self.predeclared_local_scopes.append(set())
        self.scope_kinds.append("function")
        self._record_function_arguments(node.args)
        self.predeclared_local_scopes[-1].update(_lambda_local_bindings(node))
        self.visit(node.body)
        self.scope_kinds.pop()
        self.predeclared_local_scopes.pop()
        self.index_name_scopes.pop()
        self.import_alias_scopes.pop()
        self.scopes.pop()

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._record_target_bindings(node.target)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._record_target_bindings(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        restore_binding = None
        if node.name is not None:
            restore_binding = self._temporary_name_binding(node.name)
            self._record_name_binding(node.name)
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            if restore_binding is not None:
                restore_binding()

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._record_target_bindings(
            node.target, scope_index=self._named_expr_binding_scope_index()
        )

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._record_target_bindings(target)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            case_bound_names = _pattern_names(case.pattern)
            restore_bindings = [self._temporary_name_binding(name) for name in case_bound_names]
            for name in case_bound_names:
                self._record_name_binding(name)
            try:
                if case.guard is not None:
                    self.visit(case.guard)
                for statement in case.body:
                    self.visit(statement)
            finally:
                for restore_binding in reversed(restore_bindings):
                    restore_binding()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_settings_assignment(target, node.value, node)
        self.visit(node.value)
        self._record_constant_assignments(node.targets, node.value)
        self._record_index_assignments(node.targets, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record_settings_assignment(node.target, node.value, node)
            self.visit(node.value)
            self._record_constant_assignments((node.target,), node.value)
            self._record_index_assignments((node.target,), node.value)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        resolved_call_name = self._resolve_name(call_name)
        class_name = resolved_call_name.rsplit(".", 1)[-1]
        chunking_factory_strategy = _chunking_factory_strategy(resolved_call_name)

        if _is_supported_call(resolved_call_name, "QdrantVectorStore"):
            self._record_qdrant_vector_store(node)
        elif _is_supported_call(resolved_call_name, "OpenAIEmbedding") or _is_supported_call(
            resolved_call_name, "AzureOpenAIEmbedding"
        ):
            self._record_openai_embedding(node, provider=_embedding_provider(class_name))
        elif _is_supported_call(resolved_call_name, "HuggingFaceEmbedding") or _is_supported_call(
            resolved_call_name, "SentenceTransformerEmbedding"
        ):
            self._record_named_embedding(node, provider=_embedding_provider(class_name))
        elif any(
            _is_supported_call(resolved_call_name, supported_class)
            for supported_class in {"SentenceSplitter", "TokenTextSplitter", "SimpleNodeParser"}
        ):
            self._record_chunking(node, strategy=_chunking_strategy(class_name))
        elif chunking_factory_strategy is not None:
            self._record_chunking(node, strategy=chunking_factory_strategy)
        elif _is_supported_call(resolved_call_name, "VectorIndexRetriever") or (
            call_name.endswith(".as_retriever") and self._is_llamaindex_index_receiver(call_name)
        ):
            self._record_retriever(node)
        elif _is_supported_retriever_query_engine_factory(resolved_call_name):
            self._record_retriever(node)
        elif any(
            _is_supported_call(resolved_call_name, supported_class)
            for supported_class in {"SentenceTransformerRerank", "CohereRerank", "LLMRerank"}
        ):
            self._record_reranker(node, provider=_reranker_provider(class_name))
        else:
            self._warn_on_unverified_supported_call(call_name, resolved_call_name, node)

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

        model = self._keyword_value(node, "model", "model_name")
        if provider == "azure_openai":
            deployment = self._keyword_value(node, "deployment_name", "engine")
            if model is _MISSING and deployment is not _MISSING:
                self.warnings.append(
                    "Ignored embedding.model at "
                    f"{self._location(node)}: Azure deployment_name/engine is not the "
                    "underlying embedding model; fill embedding.model manually."
                )
        elif model is _MISSING and node.args:
            model = _literal_value(node.args[0], self._constants())
        self._add_string("embedding.model", model, node)

        dimensions = self._keyword_value(node, "dimensions", "embed_dim")
        if dimensions is _MISSING and isinstance(model, str):
            dimensions = OPENAI_EMBEDDING_DIMENSIONS.get(model, _MISSING)
        self._add_positive_int("embedding.dimensions", dimensions, node)

    def _record_named_embedding(self, node: ast.Call, *, provider: str) -> None:
        self._add_string("embedding.provider", provider, node)
        model = self._keyword_value(node, "model_name", "model")
        if model is _MISSING and node.args:
            model = _literal_value(node.args[0], self._constants())
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
        if provider is None:
            self._add_value("retriever.reranker.provider", None, node)
        else:
            self._add_string("retriever.reranker.provider", provider, node)
        model = self._keyword_value(node, "model", "model_name")
        self._add_string("retriever.reranker.model", model, node)

    def _record_settings_assignment(
        self, target: ast.expr, value_node: ast.expr, location_node: ast.AST
    ) -> None:
        target_name = self._resolve_name(_attribute_name(target))
        value = _literal_value(value_node, self._constants())
        if target_name.startswith("llama_index.") and target_name.endswith("Settings.chunk_size"):
            self._add_positive_int("chunking.chunk_size", value, location_node)
            self._add_string("chunking.strategy", "llamaindex_settings", location_node)
        elif target_name.startswith("llama_index.") and target_name.endswith(
            "Settings.chunk_overlap"
        ):
            self._add_non_negative_int("chunking.chunk_overlap", value, location_node)
            self._add_string("chunking.strategy", "llamaindex_settings", location_node)

    def _keyword_value(self, node: ast.Call, *names: str) -> Any:
        for keyword in node.keywords:
            if keyword.arg in names:
                return _literal_value(keyword.value, self._constants())
        return _MISSING

    def _record_constant_assignments(
        self, targets: tuple[ast.expr, ...] | list[ast.expr], value_node: ast.expr
    ) -> None:
        value = _literal_value(value_node, self._constants())
        for target in targets:
            if isinstance(target, ast.Name):
                self._clear_import_binding(target.id)
                self.scopes[-1][target.id] = value
                continue

            self._record_target_bindings(target)

    def _record_name_binding(self, name: str, *, scope_index: int | None = None) -> None:
        if scope_index is None:
            scope_index = len(self.scopes) - 1
        self._clear_import_binding(name, scope_index=scope_index)
        self.scopes[scope_index][name] = _MISSING

    def _record_target_bindings(self, target: ast.AST, *, scope_index: int | None = None) -> None:
        for name in _target_names(target):
            self._record_name_binding(name, scope_index=scope_index)

    def _clear_import_binding(self, name: str, *, scope_index: int | None = None) -> None:
        if scope_index is None:
            scope_index = len(self.import_alias_scopes) - 1
        self.import_alias_scopes[scope_index].pop(name, None)
        self.index_name_scopes[scope_index].discard(name)

    def _record_star_import_binding(self) -> None:
        for name in tuple(self.import_alias_scopes[-1]):
            self._record_name_binding(name)

    def _temporary_name_binding(self, name: str):
        scope_index = len(self.scopes) - 1
        scope_had_name = name in self.scopes[scope_index]
        previous_scope_value = self.scopes[scope_index].get(name)
        import_had_name = name in self.import_alias_scopes[scope_index]
        previous_import_value = self.import_alias_scopes[scope_index].get(name)
        index_had_name = name in self.index_name_scopes[scope_index]

        def restore() -> None:
            if scope_had_name:
                self.scopes[scope_index][name] = previous_scope_value
            else:
                self.scopes[scope_index].pop(name, None)

            if import_had_name:
                self.import_alias_scopes[scope_index][name] = previous_import_value
            else:
                self.import_alias_scopes[scope_index].pop(name, None)

            if index_had_name:
                self.index_name_scopes[scope_index].add(name)
            else:
                self.index_name_scopes[scope_index].discard(name)

        return restore

    def _record_index_assignments(
        self, targets: tuple[ast.expr, ...] | list[ast.expr], value_node: ast.expr
    ) -> None:
        is_index_value = False
        if isinstance(value_node, ast.Call):
            is_index_value = _is_llamaindex_index_call(
                self._resolve_name(_call_name(value_node.func))
            )

        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if is_index_value:
                self.index_name_scopes[-1].add(target.id)
            else:
                self.index_name_scopes[-1].discard(target.id)

    def _visit_nested_scope(self, node: ast.AST, *, kind: str) -> None:
        self.scopes.append({})
        self.import_alias_scopes.append({})
        self.index_name_scopes.append(set())
        self.predeclared_local_scopes.append(set())
        self.scope_kinds.append(kind)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._record_function_arguments(node.args)
            self.predeclared_local_scopes[-1].update(_function_local_bindings(node))
        self.generic_visit(node)
        self.scope_kinds.pop()
        self.predeclared_local_scopes.pop()
        self.index_name_scopes.pop()
        self.import_alias_scopes.pop()
        self.scopes.pop()

    def _visit_comprehension(
        self, generators: list[ast.comprehension], result_nodes: tuple[ast.AST, ...]
    ) -> None:
        self.scopes.append({})
        self.import_alias_scopes.append({})
        self.index_name_scopes.append(set())
        self.predeclared_local_scopes.append(set())
        self.scope_kinds.append("comprehension")

        for generator in generators:
            for name in _target_names(generator.target):
                self.scopes[-1][name] = _MISSING

        for generator in generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for result_node in result_nodes:
            self.visit(result_node)

        self.scope_kinds.pop()
        self.predeclared_local_scopes.pop()
        self.index_name_scopes.pop()
        self.import_alias_scopes.pop()
        self.scopes.pop()

    def _record_function_arguments(self, arguments: ast.arguments) -> None:
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            self.scopes[-1][argument.arg] = _MISSING
        if arguments.vararg is not None:
            self.scopes[-1][arguments.vararg.arg] = _MISSING
        if arguments.kwarg is not None:
            self.scopes[-1][arguments.kwarg.arg] = _MISSING

    def _constants(self) -> dict[str, Any]:
        constants: dict[str, Any] = {}
        current_class_scope_index = None
        if self.scope_kinds[-1] == "class":
            current_class_scope_index = len(self.scope_kinds) - 1

        for index, (kind, scope) in enumerate(zip(self.scope_kinds, self.scopes, strict=True)):
            if kind == "class" and index != current_class_scope_index:
                continue
            constants.update(scope)
        return constants

    def _record_import_alias(self, local_name: str, qualified_name: str) -> None:
        self.scopes[-1].pop(local_name, None)
        self.import_alias_scopes[-1][local_name] = qualified_name

    def _resolve_name(self, name: str) -> str:
        if not name:
            return ""

        parts = name.split(".")
        root = parts[0]
        current_class_scope_index = None
        if self.scope_kinds[-1] == "class":
            current_class_scope_index = len(self.scope_kinds) - 1

        for index in range(len(self.scope_kinds) - 1, -1, -1):
            kind = self.scope_kinds[index]
            if kind == "class" and index != current_class_scope_index:
                continue

            qualified_root = self.import_alias_scopes[index].get(root)
            if qualified_root is not None:
                return ".".join((qualified_root, *parts[1:]))

            if root in self.scopes[index] or root in self.predeclared_local_scopes[index]:
                return name

        return name

    def _is_llamaindex_index_receiver(self, call_name: str) -> bool:
        receiver = call_name.rsplit(".as_retriever", 1)[0]
        if "." in receiver:
            return False
        return self._nearest_binding_is_index(receiver)

    def _nearest_binding_is_index(self, name: str) -> bool:
        current_class_scope_index = None
        if self.scope_kinds[-1] == "class":
            current_class_scope_index = len(self.scope_kinds) - 1

        for index in range(len(self.scope_kinds) - 1, -1, -1):
            kind = self.scope_kinds[index]
            if kind == "class" and index != current_class_scope_index:
                continue

            if name in self.scopes[index] or name in self.index_name_scopes[index]:
                return name in self.index_name_scopes[index]

        return False

    def _named_expr_binding_scope_index(self) -> int:
        for index in range(len(self.scope_kinds) - 1, -1, -1):
            if self.scope_kinds[index] != "comprehension":
                return index
        return len(self.scope_kinds) - 1

    def _warn_on_unverified_supported_call(
        self, call_name: str, resolved_call_name: str, node: ast.AST
    ) -> None:
        if _supported_call_class_name(call_name) is None:
            return
        if resolved_call_name.startswith("llama_index."):
            return
        self.warnings.append(
            f"Ignored {call_name} at {self._location(node)}: import could not be verified as "
            "a supported LlamaIndex + Qdrant pattern."
        )

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
    unique_values: dict[str, DiscoveredValue] = {}
    for discovered_value in discovered_values:
        key = json.dumps(discovered_value.value, sort_keys=True)
        unique_values.setdefault(key, discovered_value)

    if len(unique_values) > 1:
        alternate_values = [
            f"also found {value.value!r} at {value.location}"
            for key, value in unique_values.items()
            if key != json.dumps(first.value, sort_keys=True)
        ]
        warnings.append(
            f"Multiple values found for {field_path}; using {first.value!r} from "
            f"{first.location}; {'; '.join(alternate_values)}."
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


class _FunctionLocalBindingCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.outer_scope_names: set[str] = set()

    def _add_name(self, name: str) -> None:
        if name not in self.outer_scope_names:
            self.names.add(name)

    def _add_names(self, names: set[str]) -> None:
        self.names.update(names - self.outer_scope_names)

    def _record_outer_scope_names(self, names: list[str]) -> None:
        self.outer_scope_names.update(names)
        self.names.difference_update(names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_name(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_name(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_name(node.name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add_name(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self._add_name(alias.asname or alias.name)

    def visit_Global(self, node: ast.Global) -> None:
        self._record_outer_scope_names(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._record_outer_scope_names(node.names)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._add_names(_target_names(target))
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._add_names(_target_names(node.target))
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._add_names(_target_names(node.target))
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._add_names(_target_names(node.target))
        self.visit(node.iter)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._add_names(_target_names(item.optional_vars))
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        for statement in node.body:
            self.visit(statement)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._add_names(_target_names(node.target))
        self.visit(node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._add_names(_target_names(target))

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            if case.guard is not None:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _function_local_bindings(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    collector = _FunctionLocalBindingCollector()
    for statement in node.body:
        collector.visit(statement)
    return collector.names


def _lambda_local_bindings(node: ast.Lambda) -> set[str]:
    collector = _FunctionLocalBindingCollector()
    collector.visit(node.body)
    return collector.names


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in node.elts:
            names.update(_target_names(element))
        return names
    return set()


def _pattern_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.MatchAs):
        names = _pattern_names(node.pattern) if node.pattern is not None else set()
        if node.name is not None:
            names.add(node.name)
        return names
    if isinstance(node, ast.MatchStar):
        return {node.name} if node.name is not None else set()
    if isinstance(node, ast.MatchMapping):
        names: set[str] = set()
        for pattern in node.patterns:
            names.update(_pattern_names(pattern))
        if node.rest is not None:
            names.add(node.rest)
        return names
    if isinstance(node, ast.MatchSequence):
        names: set[str] = set()
        for pattern in node.patterns:
            names.update(_pattern_names(pattern))
        return names
    if isinstance(node, ast.MatchClass):
        names: set[str] = set()
        for pattern in (*node.patterns, *node.kwd_patterns):
            names.update(_pattern_names(pattern))
        return names
    if isinstance(node, ast.MatchOr):
        names: set[str] = set()
        for pattern in node.patterns:
            names.update(_pattern_names(pattern))
        return names
    return set()


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
        if _is_supported_call(
            call_name.removesuffix(".from_defaults"), class_name
        ) and call_name.endswith(".from_defaults"):
            return _chunking_strategy(class_name)
    return None


def _is_supported_call(call_name: str, class_name: str) -> bool:
    return any(
        call_name == f"llama_index{suffix}" for suffix in SUPPORTED_CLASS_SUFFIXES[class_name]
    )


def _is_llamaindex_index_call(call_name: str) -> bool:
    if not call_name.startswith("llama_index."):
        return False

    index_call_name = call_name
    for factory_method in SUPPORTED_INDEX_FACTORY_METHODS:
        factory_call_name = call_name.removesuffix(f".{factory_method}")
        if factory_call_name != call_name:
            index_call_name = factory_call_name
            break
    return any(index_call_name == f"llama_index{suffix}" for suffix in SUPPORTED_INDEX_SUFFIXES)


def _is_supported_retriever_query_engine_factory(call_name: str) -> bool:
    if not call_name.endswith(".from_args"):
        return False

    query_engine_name = call_name.removesuffix(".from_args")
    return any(
        query_engine_name == f"llama_index{suffix}"
        for suffix in SUPPORTED_RETRIEVER_QUERY_ENGINE_SUFFIXES
    )


def _supported_call_class_name(call_name: str) -> str | None:
    parts = call_name.split(".")
    candidates = [parts[-1]]
    if parts[-1] == "from_defaults" and len(parts) >= 2:
        candidates.append(parts[-2])

    for candidate in candidates:
        if candidate in SUPPORTED_CLASS_SUFFIXES:
            return candidate
    return None


def _reranker_provider(class_name: str) -> str | None:
    providers = {
        "CohereRerank": "cohere",
        "SentenceTransformerRerank": "sentence_transformers",
        "LLMRerank": None,
    }
    return providers[class_name]
