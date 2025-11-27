from abc import ABC, abstractmethod

class IKnowledgeBase(ABC):

    @abstractmethod
    def lookup(self, key: str) -> dict:
        pass

    @abstractmethod
    def store_fact(self, key: str, value: dict):
        pass

    @abstractmethod
    def get_medical_rules(self) -> dict:
        pass
