from abc import ABC, abstractmethod

class IAgentState(ABC):

    @abstractmethod
    def get_state(self) -> dict:
        pass

    @abstractmethod
    def update_state(self, data: dict):
        pass
