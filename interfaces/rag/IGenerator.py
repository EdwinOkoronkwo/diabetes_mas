from abc import ABC, abstractmethod

class IGenerator(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def summarize(self, text: str) -> str:
        pass
