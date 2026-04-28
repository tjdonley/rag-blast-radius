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
    assert any(
        "Multiple values found for embedding.dimensions" in warning for warning in scan.warnings
    )


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


def test_llamaindex_qdrant_cli_writes_partial_manifest(tmp_path) -> None:
    source = tmp_path / "rag_app.py"
    output = tmp_path / ".rag-manifest.partial.json"
    source.write_text(
        """
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

OpenAIEmbedding(model="text-embedding-ada-002")
SentenceSplitter(chunk_size=800, chunk_overlap=100)
QdrantVectorStore(collection_name="support_docs", enable_hybrid=False)
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
