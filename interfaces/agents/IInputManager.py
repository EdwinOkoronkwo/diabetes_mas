from abc import ABC, abstractmethod

class IInputManager(ABC):

    @abstractmethod
    def validate_input(self, user_input: str) -> dict:
        pass

    @abstractmethod
    def augment_input(self, user_input: str) -> dict:
        pass
