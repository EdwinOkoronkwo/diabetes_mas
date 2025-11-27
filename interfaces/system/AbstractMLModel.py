from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

class AbstractMLModel(ABC):
    """Interface for any machine learning model used for objective risk scoring."""
    @abstractmethod
    def predict_proba(self, texts: List[str]) -> List[float]:
        """
        Processes text input and returns the probability score (0.0 to 1.0).
        Must be implemented by concrete classes (e.g., PyTorch, TensorFlow wrappers).
        """
        pass