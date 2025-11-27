from abc import ABC, abstractmethod

class IReflectionAgent(ABC):

    @abstractmethod
    def review_output(self, output: dict) -> dict:
        pass

    @abstractmethod
    def check_consistency(self, output: dict) -> bool:
        pass

    @abstractmethod
    def safety_filter(self, output: dict) -> dict:
        pass
