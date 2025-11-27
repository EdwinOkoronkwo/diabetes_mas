# diabetes_mas/rag/MockRetriever.py (New File for Testing)

# diabetes_mas/rag/MockRetriever.py

from interfaces.rag.IRetriever import IRetriever
from interfaces.rag.IGenerator import IGenerator  # Only needed if RetrievalAgent uses it


class MockRetriever(IRetriever):
    """
    A placeholder implementation of IRetriever for development/testing
    the pipeline flow without a live vector database.
    """

    def embed_query(self, query: str) -> list[float]:
        # Mocks the embedding process
        print(f"[MockRetriever]: Embedding query: '{query[:30]}...'")
        return [1.0, 0.5, 0.0]  # Simple mock vector

    def similarity_search(self, embedded_query: list[float], k: int = 5) -> str:
        """
        Mocks the database lookup and returns a relevant context string.
        """
        print("[MockRetriever]: Performing similarity search...")

        # Simple conditional logic based on the query for a slightly better mock
        if embedded_query[0] > 0.9:  # Assuming high relevance based on mock vector
            return (
                "**Retrieved Context (T2D):** Patient's symptoms (high blood sugar, "
                "fatigue) suggest Type 2 Diabetes (T2D). Key risk factors include BMI > 30 "
                "and history of hypertension. Diagnostic steps require Fasting Plasma Glucose "
                "or HbA1c testing."
            )
        else:
            return "**Retrieved Context (General):** General wellness advice."