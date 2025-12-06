from abc import ABC, abstractmethod
from typing import Dict, Any

class IReflectionAgent(ABC):
    """
    Interface for the Reflection Agent.
    Responsible for collecting user feedback and analyzing it.
    """

    @abstractmethod
    def log_feedback(self, rating: int, message: str) -> bool:
        """
        Logs user feedback to a persistent storage.
        
        Args:
            rating (int): User rating (1-5).
            message (str): User feedback message.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def analyze_feedback(self) -> Dict[str, Any]:
        """
        Analyzes the collected feedback.
        This represents the 'background job' task.
        
        Returns:
            Dict[str, Any]: Summary of feedback analysis.
        """
        pass
