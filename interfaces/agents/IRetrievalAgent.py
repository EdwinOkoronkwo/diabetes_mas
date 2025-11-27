from abc import ABC, abstractmethod

class IRetrievalAgent(ABC):

    @abstractmethod
    def retrieve_context(self, query: str) -> dict:
        pass

    @abstractmethod
    def summarize_context(self, context: dict) -> dict:
        pass

    @abstractmethod
    @abstractmethod
    def query(self, query: str, pdf_folder: str = None) -> Dict[str, Any]:
        """
        Runs the full RAG pipeline and returns:
        {
            "answer": str,
            "summary": str,
            "retrieved_docs": List[str]
        }
        """
        pass
