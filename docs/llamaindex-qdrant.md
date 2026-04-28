# LlamaIndex + Qdrant Integration

The first integration is intentionally narrow: it inspects Python source for common LlamaIndex + Qdrant configuration patterns and generates a partial manifest draft.

It does not import your application, contact Qdrant, inspect live indexes, perform migrations, or claim that the generated file is complete.

## Usage

```bash
rag-blast integrations llamaindex-qdrant --source src/rag_app.py --output .rag-manifest.partial.json
```

`--source` can point to a Python file or a directory. Directory scans skip hidden directories, virtual environments, and common tool caches.

Use `--force` to overwrite an existing output file:

```bash
rag-blast integrations llamaindex-qdrant --source src --output .rag-manifest.partial.json --force
```

## Supported Patterns

The scanner currently recognizes simple literal or constant-backed values in these patterns:

```python
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
```

The same sample is available at `docs/examples/llamaindex_qdrant_config.py` for local smoke checks.

It can extract:

- OpenAI, Azure OpenAI, Hugging Face, and sentence-transformer embedding model names
- Default dimensions for known OpenAI embedding models
- `SentenceSplitter`, `TokenTextSplitter`, `SimpleNodeParser`, `.from_defaults`, and `Settings` chunk sizes and overlaps
- Qdrant collection names and `enable_hybrid` flags
- Retriever `similarity_top_k`, `top_k`, simple query mode, and simple reranker model patterns

## Partial Manifest Output

For the source above, the generated partial manifest looks like this:

```json
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-large",
    "dimensions": 3072
  },
  "chunking": {
    "strategy": "sentence_splitter",
    "chunk_size": 512,
    "chunk_overlap": 64
  },
  "vector_store": {
    "provider": "qdrant",
    "collection": "support_docs_v4"
  },
  "retriever": {
    "top_k": 8,
    "hybrid": true
  }
}
```

The command also prints warnings similar to:

```text
Manual review required:
- app is required and must be filled manually.
- environment is required and must be filled manually.
- caches are not inferred; add semantic cache entries manually if used.
- evals are not inferred; add retrieval eval paths manually if available.
```

Fill the missing fields before using the draft with `rag-blast check`.

## Limits

The scanner intentionally avoids false confidence:

- It only scans Python source; it does not execute code.
- Dynamic values without literal defaults are not resolved.
- If multiple values are found for a field, it uses the first deterministic source-order value and prints a warning.
- Semantic caches and retrieval evals are not inferred.
- The output is a partial manifest draft, not proof that the application is safe to deploy.
