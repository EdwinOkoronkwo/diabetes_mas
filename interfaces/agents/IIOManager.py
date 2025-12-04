from abc import ABC, abstractmethod

class IIOManager(ABC):

    @abstractmethod
    def validate_input(self, state: dict) -> dict:
        pass

    @abstractmethod
    def augment_input(self, state: dict) -> dict:
        pass
