from abc import ABC, abstractmethod

class IIOManager(ABC):

    @abstractmethod
    def validate_input(self, user_input: str) -> dict:
        pass

    @abstractmethod
    def augment_input(self, user_input: str) -> dict:
        pass
