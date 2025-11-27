from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class ILLMGenerator(ABC):
    """
    Abstract Base Class defining the contract for all LLM implementations
    in the Multi-Agent System (MAS).
    """

    @abstractmethod
    def __init__(self, model_name: str, temperature: float = 0.0):
        """Initializes the LLM with a specific model and configuration."""
        pass

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """
        Invokes the LLM to generate a text response based on a list of messages.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "Query"}]).

        Returns:
            The raw text content of the LLM's response.
        """
        pass

    @abstractmethod
    def invoke_structured(self, messages: List[Dict[str, str]], response_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes the LLM to generate a structured (JSON) response.

        Args:
            messages: A list of message dictionaries.
            response_schema: A dictionary defining the required JSON schema.

        Returns:
            The parsed JSON dictionary response.
        """
        pass