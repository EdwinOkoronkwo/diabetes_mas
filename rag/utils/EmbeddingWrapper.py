# rag/utils/EmbeddingWrapper.py
from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerEmbeddingFunction:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents (used by add_texts in ChromaIndexer)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embeds a single query string (used by similarity search)
        """
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
