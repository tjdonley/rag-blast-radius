import json
from pathlib import Path

from typer.testing import CliRunner

from rag_blast.cli import app
from rag_blast.integrations import scan_llamaindex_qdrant

runner = CliRunner()


def test_llamaindex_qdrant_scan_extracts_known_config(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

COLLECTION = "support_docs_v4"

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
vector_store = QdrantVectorStore(collection_name=COLLECTION, enable_hybrid=True)
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=8)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest == {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-large",
            "dimensions": 3072,
        },
        "chunking": {
            "strategy": "sentence_splitter",
            "chunk_size": 512,
            "chunk_overlap": 64,
        },
        "vector_store": {
            "provider": "qdrant",
            "collection": "support_docs_v4",
        },
        "retriever": {
            "top_k": 8,
            "hybrid": True,
        },
    }
    assert "app is required and must be filled manually." in scan.warnings
    assert "environment is required and must be filled manually." in scan.warnings
    assert "caches are not inferred; add semantic cache entries manually if used." in scan.warnings
    assert (
        "evals are not inferred; add retrieval eval paths manually if available." in scan.warnings
    )


def test_llamaindex_qdrant_scan_surfaces_missing_fields(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(collection_name="support_docs")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest == {
        "vector_store": {
            "provider": "qdrant",
            "collection": "support_docs",
        },
    }
    for field_path in (
        "embedding.provider",
        "embedding.model",
        "embedding.dimensions",
        "chunking.strategy",
        "chunking.chunk_size",
        "chunking.chunk_overlap",
        "retriever.top_k",
        "retriever.hybrid",
    ):
        assert f"{field_path} is required and must be filled manually." in scan.warnings


def test_llamaindex_qdrant_scan_warns_on_conflicting_values(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

first = OpenAIEmbedding(model="text-embedding-3-small")
second = OpenAIEmbedding(model="text-embedding-3-large")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["embedding"]["model"] == "text-embedding-3-small"
    assert scan.manifest["embedding"]["dimensions"] == 1536
    assert any("Multiple values found for embedding.model" in warning for warning in scan.warnings)
    assert any("also found 'text-embedding-3-large'" in warning for warning in scan.warnings)
    assert any(
        "Multiple values found for embedding.dimensions" in warning for warning in scan.warnings
    )


def test_llamaindex_qdrant_scan_resolves_supported_import_aliases(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore as QVS

vector_store = QVS(collection_name="support_docs")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["vector_store"] == {
        "provider": "qdrant",
        "collection": "support_docs",
    }


def test_llamaindex_qdrant_scan_rejects_unsupported_nested_modules(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.experimental.vector_stores.qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(collection_name="experimental")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "vector_store" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_non_llama_import_rebinding(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding
from local_embeddings import OpenAIEmbedding

embedding = OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_rejects_relative_llamaindex_imports(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from .llama_index.embeddings.openai import OpenAIEmbedding

embedding = OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_star_import_rebinding(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding
from local_embeddings import *

embedding = OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_ignores_llamaindex_star_imports(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import *

embedding = OpenAIEmbedding(model="text-embedding-3-small")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_ignores_unverified_class_names(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
class QdrantVectorStore:
    def __init__(self, collection_name):
        self.collection_name = collection_name

vector_store = QdrantVectorStore(collection_name="not_llamaindex")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "vector_store" not in scan.manifest
    assert any("Ignored QdrantVectorStore" in warning for warning in scan.warnings)
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_module_redefinition_after_import(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

class QdrantVectorStore:
    pass

vector_store = QdrantVectorStore(collection_name="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "vector_store" not in scan.manifest
    assert any("Ignored QdrantVectorStore" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_local_shadowing_after_import(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

def build_vector_store(QdrantVectorStore):
    return QdrantVectorStore(collection_name="not_llamaindex")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "vector_store" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_reads_rhs_before_assignment_shadowing(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

OpenAIEmbedding = OpenAIEmbedding(model="text-embedding-3-small")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["embedding"] == {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    }


def test_llamaindex_qdrant_scan_respects_later_function_local_bindings(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

def configure():
    OpenAIEmbedding(model="text-embedding-3-small")
    OpenAIEmbedding = object
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_comprehension_target_shadowing(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

embeddings = [OpenAIEmbedding(model="local") for OpenAIEmbedding in factories]
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_lambda_parameter_shadowing(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

factory = lambda OpenAIEmbedding: OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_loop_target_shadowing(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

for OpenAIEmbedding in factories:
    OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_starred_assignment_shadowing(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

*OpenAIEmbedding, other = row
OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_with_target_shadowing(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

with manager() as QdrantVectorStore:
    QdrantVectorStore(collection_name="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "vector_store" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_exception_target_shadowing(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

try:
    raise RuntimeError
except RuntimeError as OpenAIEmbedding:
    OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_match_capture_shadowing(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

match item:
    case OpenAIEmbedding:
        OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_comprehension_walrus_binding(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

values = [(OpenAIEmbedding := factory) for factory in factories]
OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_respects_deleted_import_binding(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

del OpenAIEmbedding
OpenAIEmbedding(model="local")
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_predeclares_function_delete_bindings(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

def configure():
    OpenAIEmbedding(model="local")
    del OpenAIEmbedding
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_predeclares_lambda_walrus_bindings(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

factory = lambda: (OpenAIEmbedding(model="local"), (OpenAIEmbedding := object))
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_predeclares_function_match_bindings(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.openai import OpenAIEmbedding

def configure(item):
    OpenAIEmbedding(model="text-embedding-3-small")
    match item:
        case OpenAIEmbedding:
            pass
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "embedding" not in scan.manifest
    assert any("No supported LlamaIndex + Qdrant" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_only_records_tracked_index_as_retriever(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.core import Settings

search.as_retriever(similarity_top_k=99)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "retriever" not in scan.manifest
    assert "retriever.top_k is required and must be filled manually." in scan.warnings


def test_llamaindex_qdrant_scan_respects_shadowed_index_receiver(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_vector_store(vector_store)

def configure(index):
    return index.as_retriever(similarity_top_k=99)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "retriever" not in scan.manifest
    assert "retriever.top_k is required and must be filled manually." in scan.warnings


def test_llamaindex_qdrant_scan_records_as_retriever_from_vector_store_index(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=8)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["retriever"]["top_k"] == 8


def test_llamaindex_qdrant_scan_rejects_similarly_named_index_classes(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.core import NotVectorStoreIndex

index = NotVectorStoreIndex()
retriever = index.as_retriever(similarity_top_k=99)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "retriever" not in scan.manifest
    assert "retriever.top_k is required and must be filled manually." in scan.warnings


def test_llamaindex_qdrant_scan_records_supported_retriever_query_engine(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever, similarity_top_k=8)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["retriever"]["top_k"] == 8


def test_llamaindex_qdrant_scan_rejects_unsupported_retriever_query_engine(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.experimental import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever, similarity_top_k=99)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert "retriever" not in scan.manifest
    assert "retriever.top_k is required and must be filled manually." in scan.warnings


def test_llamaindex_qdrant_scan_does_not_treat_azure_deployment_as_model(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

embedding = AzureOpenAIEmbedding(deployment_name="prod-embedding", dimensions=1536)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["embedding"] == {
        "provider": "azure_openai",
        "dimensions": 1536,
    }
    assert "embedding.model is required and must be filled manually." in scan.warnings
    assert any("Azure deployment_name/engine" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_counts_only_parsed_files(tmp_path) -> None:
    good_source = tmp_path / "good.py"
    bad_source = tmp_path / "bad.py"
    good_source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(collection_name="support_docs")
""".lstrip(),
        encoding="utf-8",
    )
    bad_source.write_text("def broken(:\n", encoding="utf-8")

    scan = scan_llamaindex_qdrant(tmp_path)

    assert scan.scanned_files == [good_source]
    assert any(f"Skipped {bad_source}" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_resolves_constants_in_lexical_scope(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

COLLECTION = "prod_docs"

def helper() -> None:
    COLLECTION = "test_docs"

vector_store = QdrantVectorStore(collection_name=COLLECTION)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["vector_store"]["collection"] == "prod_docs"
    assert not any("test_docs" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_resolves_function_local_constants(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

COLLECTION = "prod_docs"

def build_vector_store():
    COLLECTION = "test_docs"
    return QdrantVectorStore(collection_name=COLLECTION)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["vector_store"]["collection"] == "test_docs"


def test_llamaindex_qdrant_scan_does_not_use_outer_constant_for_dynamic_local(
    tmp_path,
) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

COLLECTION = "prod_docs"

def build_vector_store(COLLECTION):
    return QdrantVectorStore(collection_name=COLLECTION)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["vector_store"] == {"provider": "qdrant"}
    assert "vector_store.collection is required and must be filled manually." in scan.warnings


def test_llamaindex_qdrant_scan_does_not_use_class_body_constants_in_methods(
    tmp_path,
) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

COLLECTION = "prod_docs"

class Builder:
    COLLECTION = "test_docs"

    def build(self):
        return QdrantVectorStore(collection_name=COLLECTION)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["vector_store"]["collection"] == "prod_docs"


def test_llamaindex_qdrant_scan_uses_class_body_constants_in_class_body(
    tmp_path,
) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.vector_stores.qdrant import QdrantVectorStore

COLLECTION = "prod_docs"

class Builder:
    COLLECTION = "test_docs"
    vector_store = QdrantVectorStore(collection_name=COLLECTION)
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["vector_store"]["collection"] == "test_docs"


def test_llamaindex_qdrant_scan_handles_factory_calls_and_incomplete_rerankers(
    tmp_path,
) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.postprocessor.cohere_rerank import CohereRerank

parser = SimpleNodeParser.from_defaults(chunk_size=400, chunk_overlap=40)
reranker = CohereRerank()
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["chunking"] == {
        "strategy": "simple_node_parser",
        "chunk_size": 400,
        "chunk_overlap": 40,
    }
    assert scan.manifest["retriever"]["reranker"] == {"provider": "cohere"}
    assert any("retriever.reranker.model is required" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_scan_preserves_unknown_provider_reranker_detection(
    tmp_path,
) -> None:
    source = tmp_path / "rag_app.py"
    source.write_text(
        """
from llama_index.core.postprocessor import LLMRerank

reranker = LLMRerank()
""".lstrip(),
        encoding="utf-8",
    )

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["retriever"]["reranker"] == {"provider": None}
    assert any("retriever.reranker.model is required" in warning for warning in scan.warnings)


def test_llamaindex_qdrant_cli_writes_partial_manifest(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    output = tmp_path / ".rag-manifest.partial.json"
    source.write_text(
        """
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

OpenAIEmbedding(model="text-embedding-ada-002")
SentenceSplitter(chunk_size=800, chunk_overlap=100)
vector_store = QdrantVectorStore(collection_name="support_docs", enable_hybrid=False)
index = VectorStoreIndex.from_vector_store(vector_store)
index.as_retriever(similarity_top_k=8)
""".lstrip(),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "integrations",
            "llamaindex-qdrant",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["embedding"] == {
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "dimensions": 1536,
    }
    assert "Wrote partial manifest" in result.output
    assert "Manual review required" in result.output


def test_llamaindex_qdrant_cli_refuses_to_overwrite_existing_output(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    output = tmp_path / ".rag-manifest.partial.json"
    source.write_text("QdrantVectorStore(collection_name='support_docs')\n", encoding="utf-8")
    output.write_text("existing", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "integrations",
            "llamaindex-qdrant",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert output.read_text(encoding="utf-8") == "existing"


def test_integration_docs_cover_partial_manifest_behavior() -> None:
    docs = (Path(__file__).parent.parent / "docs/llamaindex-qdrant.md").read_text(encoding="utf-8")
    readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")

    assert "rag-blast integrations llamaindex-qdrant" in docs
    assert "partial manifest" in docs
    assert "Manual review required" in docs
    assert "docs/examples/llamaindex_qdrant_config.py" in docs
    assert "docs/llamaindex-qdrant.md" in readme


def test_documented_integration_example_generates_partial_manifest() -> None:
    source = Path(__file__).parent.parent / "docs/examples/llamaindex_qdrant_config.py"

    scan = scan_llamaindex_qdrant(source)

    assert scan.manifest["embedding"]["model"] == "text-embedding-3-large"
    assert scan.manifest["vector_store"]["collection"] == "support_docs_v4"
    assert scan.manifest["retriever"] == {"top_k": 8, "hybrid": True}
