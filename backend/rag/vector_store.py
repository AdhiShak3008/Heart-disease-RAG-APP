"""Qdrant vector store."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.rag.config import COLLECTION_NAME, QDRANT_PATH


class VectorStore:

    def __init__(self, vector_size: int):

        self.client = QdrantClient(path=str(QDRANT_PATH))

        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if COLLECTION_NAME not in names:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(self, ids, embeddings, payloads):

        points = []

        for pid, emb, payload in zip(ids, embeddings, payloads):
            points.append(
                PointStruct(
                    id=pid,
                    vector=emb,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    def search(self, query_vector, limit=5):

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
        )

        return response.points
