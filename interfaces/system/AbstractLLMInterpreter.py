from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

class AbstractLLMInterpreter(ABC):
    """Interface for the LLM component responsible for interpreting scores and planning."""
    @abstractmethod
    def interpret(self, symptom_text: str, ml_score: float) -> Dict[str, str]:
        """
        Takes the raw symptoms and numerical score, and uses an LLM to generate
        a dynamic risk level and structured clinical action plan.
        Returns a dictionary with 'riskLevel' (High/Moderate/Low) and 'plan'.
        """
        pass