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
