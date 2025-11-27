from abc import ABC, abstractmethod

class IPredictiveAgent(ABC):

    @abstractmethod
    def predict_risk(self, data: dict) -> dict:
        pass

    @abstractmethod
    def generate_plan(self, data: dict) -> dict:
        pass
