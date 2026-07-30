from backend.rag.embeddings import EmbeddingModel

embedder = EmbeddingModel()

vector = embedder.embed_query("What is a heart murmur?")

print(len(vector))
print(vector[:10])
