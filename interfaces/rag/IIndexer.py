from abc import ABC, abstractmethod

class IIndexer(ABC):

    @abstractmethod
    def add_document(self, document: dict):
        pass

    @abstractmethod
    def update_index(self):
        pass
