import faiss
import numpy as np
from typing import List
from .interfaces import BaseEmbeddingVectorStore
from .schemas import FaceEmbedding, MatchResult
from .exceptions import VectorStoreError

class FAISSEmbeddingVectorStore(BaseEmbeddingVectorStore):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.ids = []

    def add_vectors(self, embeddings: List[FaceEmbedding], ids: List[str]) -> None:
        try:
            vectors = np.array([embedding.vector for embedding in embeddings])
            self.index.add(vectors)
            self.ids.extend(ids)
        except Exception as e:
            raise VectorStoreError(f"Failed to add vectors: {e}")

    def search_vectors(self, query_embedding: FaceEmbedding, top_k: int) -> List[MatchResult]:
        try:
            query_vector = np.array([query_embedding.vector])
            distances, indices = self.index.search(query_vector, top_k)
            results = []
            for distance, index in zip(distances[0], indices[0]):
                if index != -1:
                    results.append(MatchResult(matched=True, confidence=1 - distance, matched_id=self.ids[index], distance=distance))
            return results
        except Exception as e:
            raise VectorStoreError(f"Failed to search vectors: {e}")