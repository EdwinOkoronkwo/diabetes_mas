from typing import List, Any, Dict, Optional
from langchain_core.documents import Document

class RagAgentState:
    """
    Internal state for the RAG pipeline.
    Fully decoupled from MAS AgentState.
    """
    def __init__(
        self,
        pdf_folder: str = "../data/01_raw",
        query: str = "",
        indexer: Any = None,
        retriever: Any = None,
        generator: Any = None,
        top_k: int = 5,
    ):
        self.pdf_folder = pdf_folder
        self.query = query
        self.top_k = top_k

        # Document storage
        self.documents: List[Document] = []
        self.chunks: List[Document] = []

        # RAG components
        self.indexer = indexer
        self.retriever = retriever
        self.generator = generator
        self.embedded_vectors: Any = None  # keep in case needed for indexer

        # Node outputs
        self.retrieved_docs: List[Document] = []
        self.answer: str = ""
        self.summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Return state as a dictionary for node compatibility."""
        return {
            "pdf_folder": self.pdf_folder,
            "query": self.query,
            "top_k": self.top_k,
            "documents": self.documents,
            "chunks": self.chunks,
            "indexer": self.indexer,
            "retriever": self.retriever,
            "generator": self.generator,
            "embedded_vectors": self.embedded_vectors,
            "retrieved_docs": self.retrieved_docs,
            "answer": self.answer,
            "summary": self.summary,
        }
