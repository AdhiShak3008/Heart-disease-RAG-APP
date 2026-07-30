"""Embeddings generation for the RAG pipeline."""

from sentence_transformers import SentenceTransformer

from backend.rag.config import EMBEDDING_MODEL


class EmbeddingModel:
    """Wrapper around SentenceTransformer."""

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return embedding.tolist()
