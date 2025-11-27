from abc import ABC, abstractmethod

class ICheckpointer(ABC):

    @abstractmethod
    def save(self, state: dict):
        pass

    @abstractmethod
    def load(self) -> dict:
        pass
