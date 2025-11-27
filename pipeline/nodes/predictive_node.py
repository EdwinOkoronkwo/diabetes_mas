# pipeline/nodes/predictive_node.py
import logging
import asyncio
from typing import Dict, Any


# Assuming these global instances are initialized elsewhere (e.g., in main.py)
# from system.agents import predictive_agent
# from system.types import AgentState

# --- Mocking the AgentState and Agent for clarity ---
class PredictiveAgent:
    """Mock for the actual PredictiveAgent class."""

    def __init__(self):
        logging.info("PredictiveAgent initialized.")

    async def predict_risk(self, rag_answer: str, retrieved_context: str = "") -> Dict[str, Any]:
        """The actual business logic method."""
        # This is where your actual async logic from the second snippet lives
        return {
            "risk": "high",
            "score": 0.95,
            "plan": {"immediateActions": ["Refer to ER"]},
            "method": "hybrid"
        }


# Global agent instance (used by the node)
predictive_agent_instance = PredictiveAgent()


# --- End Mocking ---


def predictive_node(agent_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    This function acts as the synchronous wrapper required by a LangGraph node.
    It executes the core business logic from the PredictiveAgent class.
    """
    logger = logging.getLogger("PredictiveNode")

    rag_answer = agent_state.get("rag_answer", "")
    retrieved_context = agent_state.get("retrieved_context", "")

    logger.info("Running PredictiveAgent's core logic...")

    # NOTE: You MUST use an asyncio method (like running a dedicated loop)
    # to call the async predict_risk method from a synchronous graph node.
    try:
        # Use asyncio.run or similar method to execute the async function synchronously
        # In a real LangGraph setup, the runner handles this, but here we force it.
        if asyncio.get_event_loop().is_running():
            # If already in an event loop (e.g., Jupyter, main async loop)
            # This is complex and depends heavily on the framework's runner.
            # We'll stick to the simpler linear call for demonstration:
            # result = asyncio.create_task(predictive_agent_instance.predict_risk(rag_answer, retrieved_context)).result()
            logger.warning("Agent requires async environment. Using mock result for simplicity.")
            # Fallback to a synchronous call that should ideally be wrapped
            result = asyncio.run(predictive_agent_instance.predict_risk(rag_answer, retrieved_context))
        else:
            result = asyncio.run(predictive_agent_instance.predict_risk(rag_answer, retrieved_context))

        agent_state["predictive_result"] = result

    except Exception as e:
        logger.error(f"Error executing PredictiveAgent: {e}")
        agent_state["error"] = f"PredictiveAgent error: {e}"

    return agent_state
