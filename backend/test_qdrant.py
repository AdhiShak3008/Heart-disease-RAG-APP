from backend.rag.embeddings import EmbeddingModel
from backend.rag.vector_store import VectorStore

embedder = EmbeddingModel()

vector = embedder.embed_query("What is a heart murmur?")

store = VectorStore(
    vector_size=len(vector),
)

print("Collection created successfully!")
