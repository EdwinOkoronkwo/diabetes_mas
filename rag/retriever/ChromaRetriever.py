# rag/retriever/ChromaRetriever.py
import logging
from langchain_community.vectorstores import Chroma

from rag.db.chroma_client import get_chroma_db

logger = logging.getLogger(__name__)
persist_directory=r"C:\CentennialCollege\COMP248_AI_Software_Design\Project\diabetes_mas\rag\chroma_db"

class ChromaRetriever:
    def __init__(self, collection_name: str, embedding_fn, chroma_db=None):
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        # Use provided chroma_db or create a new one
        self.db = chroma_db if chroma_db else get_chroma_db(collection_name, embedding_fn=embedding_fn)
        logger.info(f"[Retriever] Initialized for collection: {self.collection_name}")

    def similarity_search(self, query: str, top_k: int = 3):
        if self.db is None:
            logger.warning("[Retriever] Chroma DB not initialized")
            return []
        return self.db.similarity_search(query, k=top_k)

    def similarity_search_by_vector(self, vector: list[float], top_k: int = 3):
        """Search using a precomputed embedding vector."""
        if self.db is None:
            logger.warning("[Retriever] Chroma DB not initialized")
            return []
        return self.db.similarity_search_by_vector(vector, k=top_k)

    def retrieve(self, query: str = None, vector: list[float] = None, top_k: int = 3):
        """Unified retrieve method using vector if available, otherwise text query."""
        if vector is not None and isinstance(vector, list) and len(vector) > 0:
            return self.similarity_search_by_vector(vector, top_k=top_k)
        elif query:
            return self.similarity_search(query, top_k=top_k)
        else:
            logger.warning("[Retriever] No query or vector provided for retrieval")
            return []



# class ChromaRetriever:
#     def __init__(self, collection_name: str, embedding_fn, chroma_db=None):
#         self.collection_name = collection_name
#         self.embedding_fn = embedding_fn
#         self.db = chroma_db if chroma_db else get_chroma_db(collection_name, embedding_fn=embedding_fn)
#         logger.info(f"[Retriever] Initialized for collection: {self.collection_name}")
#
#     def similarity_search(self, query: str, top_k: int = 3):
#         if self.db is None:
#             logger.warning("[Retriever] Chroma DB not initialized")
#             return []
#         return self.db.similarity_search(query, k=top_k)
#
#     def retrieve(self, query: str, top_k: int = 3):
#         return self.similarity_search(query, top_k=top_k)


# class ChromaRetriever:
#     def __init__(self, collection_name: str, embedding_fn):
#         self.collection_name = collection_name
#         self.embedding_fn = embedding_fn
#         self.db = Chroma(
#             collection_name=self.collection_name,
#             embedding_function=self.embedding_fn,
#             persist_directory=r"C:\CentennialCollege\COMP248_AI_Software_Design\Project\diabetes_mas\chroma_db"
#         )
#         logger.info(f"[Retriever] ChromaRetriever loaded for collection: {self.collection_name}")
#
#     def similarity_search(self, query: str, top_k: int = 3):
#         if self.db is None:
#             logger.warning("[Retriever] Chroma DB not initialized")
#             return []
#         return self.db.similarity_search(query, k=top_k)
#
#     def similarity_search_with_score(self, query: str, top_k: int = 3):
#         if self.db is None:
#             logger.warning("[Retriever] Chroma DB not initialized")
#             return []
#         return self.db.similarity_search_with_score(query, k=top_k)
#
#     def retrieve(self, query: str, top_k: int = 3):
#         return self.similarity_search(query, top_k=top_k)




# from interfaces.rag.IRetriever import IRetriever
# import random
# from typing import List
#
# from rag.knowledge_base import KNOWLEDGE_BASE
#
#
# # In a real application, this class would connect to a Chroma vector store.
# # For simplicity and to avoid external dependencies, this version simulates the
# # process of searching through a small, hardcoded knowledge base.
#
# # Define a small, fixed knowledge base (KB) for simulation
#
#
# class ChromaRetriever(IRetriever):
#     """
#     A simulated RAG Retriever using a fixed, in-memory knowledge base.
#     It implements the IRetriever interface.
#     """
#
#     def embed_query(self, query: str) -> List[float]:
#         """
#         Simulates embedding the query. Returns a vector based on keywords.
#         """
#         # Simple simulation: return a vector based on keyword presence
#         vector = [0.0, 0.0, 0.0]
#         query_lower = query.lower()
#         if "sugar" in query_lower or "diabetes" in query_lower:
#             vector[0] = 1.0  # High relevance to T2D
#         if "fatigue" in query_lower or "symptoms" in query_lower:
#             vector[1] = 0.8  # Relevance to symptoms
#         if "diagnosis" in query_lower or "test" in query_lower:
#             vector[2] = 0.6  # Relevance to diagnosis
#
#         print(f"[ChromaRetriever]: Query Embedded. Vector: {vector}")
#         return vector
#
#     def similarity_search(self, embedded_query: List[float], k: int = 5) -> str:
#         """
#         Simulates similarity search against the fixed knowledge base.
#         Returns the top matching documents concatenated into a single context string.
#         """
#         print("[ChromaRetriever]: Simulating search for relevant documents.")
#
#         # Simple simulation: select documents where the query vector has high values
#         is_high_t2d_relevance = embedded_query[0] > 0.9
#
#         relevant_docs = []
#
#         if is_high_t2d_relevance:
#             # If T2D is relevant, pull documents related to Diagnosis and Risk
#             relevant_docs.append(KNOWLEDGE_BASE[0])  # Diagnosis
#             relevant_docs.append(KNOWLEDGE_BASE[1])  # Risk factors
#             relevant_docs.append(KNOWLEDGE_BASE[3])  # Screening
#         else:
#             # Default to a general guideline
#             relevant_docs.append(KNOWLEDGE_BASE[2])
#
#             # Compile the final context string
#         context_parts = [f"Source {doc['id']}: {doc['document']}" for doc in relevant_docs]
#         compiled_context = "\n---\n".join(context_parts)
#
#         print(f"[ChromaRetriever]: Found {len(relevant_docs)} documents.")
#
#         return compiled_context