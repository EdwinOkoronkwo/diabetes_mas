import logging
from typing import Dict, Any, List

from system.LLMGenerator import LLMGenerator

logger = logging.getLogger(__name__)

# Global instance for the generic LLM Generator
# This LLM is initialized once and can be used for any non-RAG, non-Predictive tasks.
_generic_llm_generator = None


def load_mistral_llm() -> LLMGenerator:
    """
    Initializes or returns the singleton instance of the LLM Generator.
    This fulfills the user's request for a separate LLM load function.

    NOTE: We are using LLMGenerator which wraps Gemini, serving as the system's
    generic conversational LLM.
    """
    global _generic_llm_generator
    if _generic_llm_generator is None:
        try:
            # Initialize with default settings
            _generic_llm_generator = LLMGenerator()
            logger.info("[LLM Utility] Generic LLM Generator initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize generic LLM Generator: {e}")
            # Fallback for error handling
            _generic_llm_generator = None
    return _generic_llm_generator


def llm_node(agent_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    A generic node for independent LLM calls (e.g., clarification, routing).
    It takes 'current_input' and returns a simple LLM response.
    """
    llm = load_mistral_llm()
    if not llm:
        return {"llm_output": "LLM service unavailable."}

    user_query = agent_state.get("current_input", "Hello.")

    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
        {"role": "user", "content": user_query}
    ]

    # Use the simple text invoke method
    response_text = llm.invoke(messages)

    logger.info("[LLM Node] Simple LLM response generated.")
    return {"llm_output": response_text}