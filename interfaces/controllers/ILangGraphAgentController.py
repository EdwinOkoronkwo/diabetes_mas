# interfaces/controllers/ILangGraphAgentController.py

from abc import ABC, abstractmethod
from typing import Dict


class ILangGraphAgentController(ABC):

    @abstractmethod
    def run(self, user_input: str) -> Dict:
        """Execute the full MAS + RAG pipeline and return AgentState."""
        pass
