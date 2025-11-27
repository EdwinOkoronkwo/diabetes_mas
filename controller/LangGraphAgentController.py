# controllers/LangGraphAgentController.py

from typing import Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LangGraphAgentController:
    def __init__(self, input_manager, retrieval_agent, predictive_agent):
        """
        input_manager: handles input validation/augmentation
        retrieval_agent: handles RAG retrieval (RagAgentState -> RAG answer & summary)
        predictive_agent: handles risk prediction and plan generation
        """
        self.input_manager = input_manager
        self.retrieval = retrieval_agent
        self.predictive_agent = predictive_agent

    def run(self, user_input: str) -> Dict:
        """
        Complete pipeline:
        1. Validate and augment input
        2. Run RAG retrieval to get answer & summary
        3. Pass RAG output to PredictiveAgent
        """
        # --- Step 1: Initialize state ---
        state = {
            "current_input": user_input,
            "query": user_input,
            "validated_input": None,
            "documents": [],
            "retrieved_docs": [],
            "top_k": 5,
            "rag_answer": None,
            "rag_summary": None,
            "retrieved_context": None,
            "metadata": {},
        }

        # --- Step 2: Input validation / augmentation ---
        state = self.input_manager.validate_input(state)
        state = self.input_manager.augment_input(state)

        # --- Step 3: RAG retrieval ---
        state = self.retrieval.run(state)
        logger.info(f"[Controller] RAG answer: {state.get('rag_answer')}")
        logger.info(f"[Controller] RAG summary: {state.get('rag_summary')}")

        # --- Step 4: Predictive agent uses RAG output ---
        risk_result = self.predictive_agent.predict_risk(state)
        plan_result = self.predictive_agent.generate_plan(state)

        # --- Step 5: Update state with predictive outputs ---
        state["predicted_risk"] = risk_result.get("risk")
        state["recommended_plan"] = plan_result.get("plan")

        logger.info(f"[Controller] Predicted risk: {state['predicted_risk']}")
        logger.info(f"[Controller] Recommended plan: {state['recommended_plan']}")

        return state


# class LangGraphAgentController(ILangGraphAgentController):
#     def __init__(self, input_manager, retrieval_agent):
#         self.input_manager = input_manager
#         self.retrieval = retrieval_agent
#
#     def run(self, user_input: str) -> Dict:
#         state = {
#             "current_input": user_input,
#             "query": user_input,
#             "documents": [],
#             "retrieved_docs": [],
#             "top_k": 5,
#             "rag_answer": None,
#             "rag_summary": None,
#         }
#
#         # Input pipeline
#         state = self.input_manager.validate_input(state)
#         state = self.input_manager.augment_input(state)
#
#         # Retrieval pipeline
#         state = self.retrieval.run(state)
#
#         return state


# class LangGraphAgentController(ILangGraphAgentController):
#     def __init__(self, input_manager, retrieval_agent):
#         self.input_manager = input_manager
#         self.retrieval = retrieval_agent
#
#     def run(self, user_input: str) -> Dict:
#         """Main MAS control flow."""
#
#         state = {
#             "current_input": user_input,
#             "validated_input": None,
#             "augmented_input": None,
#             "retrieved_context": None,
#             "rag_answer": None,
#             "rag_summary": None,
#             "metadata": {},
#             "embedded_query": None,
#         }
#
#         # Step 1 — Input Manager
#         state = self.input_manager.validate_input(state)
#         state = self.input_manager.augment_input(state)
#
#         # Step 2 — Retrieval Agent (calls RAG)
#         state = self.retrieval.run(state)
#
#         return state
