from abc import ABC, abstractmethod

class IRetriever(ABC):

    @abstractmethod
    def embed_query(self, text: str):
        pass

    @abstractmethod
    def similarity_search(self, embedding, k: int):
        pass
