"""Semantic retriever for Qdrant."""

from backend.rag.config import TOP_K
from backend.rag.embeddings import EmbeddingModel
from backend.rag.vector_store import VectorStore


class Retriever:
    """Retrieve relevant chunks from the vector database."""

    def __init__(self):

        self.embedder = EmbeddingModel()

        sample_vector = self.embedder.embed_query("test")

        self.store = VectorStore(
            vector_size=len(sample_vector),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks.

        Returns
        -------
        List of payload dictionaries.
        """

        query_vector = self.embedder.embed_query(query)

        results = self.store.search(
            query_vector=query_vector,
            limit=top_k,
        )

        retrieved = []

        for hit in results:

            payload = hit.payload.copy()

            payload["score"] = hit.score

            retrieved.append(payload)

        return retrieved
