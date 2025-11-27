# import logging
# from interfaces.rag.IGenerator import IGenerator
# from typing import List
#
# logger = logging.getLogger(__name__)
#
# class MockLLMGenerator(IGenerator):
#     def __init__(self, api_key: str = "mock_key", model_name: str = "mock-model"):
#         logger.info(f"[MockLLM] Initialized mock LLM: {model_name}")
#
#     def generate(self, query: str, docs: List[str]) -> str:
#         logger.info("[MockLLM] generate called")
#         return f"[MOCK ANSWER] Query: {query} | Docs: {len(docs)} retrieved"
#
#     def summarize(self, text: str) -> str:
#         logger.info("[MockLLM] summarize called")
#         return f"[MOCK SUMMARY] {text[:50]}..."  # first 50 chars

import logging
from typing import List

from interfaces.rag.IGenerator import IGenerator

logger = logging.getLogger(__name__)

class MockLLMGenerator(IGenerator):
    """Mock generator that returns static responses for testing."""

    def generate(self, query: str, docs: List[str]) -> str:
        logger.info("[MockLLM] generate called")
        # For testing, return static answer
        return f"MOCK ANSWER for query: {query}"

    def summarize(self, text: str) -> str:
        logger.info("[MockLLM] summarize called")
        # Return a static summary
        return f"MOCK SUMMARY of text length {len(text)}"

