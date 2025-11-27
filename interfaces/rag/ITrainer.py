from abc import ABC, abstractmethod

class ITrainer(ABC):

    @abstractmethod
    def fine_tune(self, training_data):
        pass

    @abstractmethod
    def evaluate(self, eval_data):
        pass
