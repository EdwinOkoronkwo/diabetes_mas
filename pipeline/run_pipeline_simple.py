# pipeline/run_pipeline.py
import asyncio
import logging
from pprint import pprint
from typing import Dict, Any

from agents.InputManager import InputManager
# --- Import Required Agents and Components ---
# These are all concrete implementations
from agents.RetrievalAgent import RetrievalAgent
from agents.PredictiveAgent import PredictiveAgent
from pipeline.build_graph import build_pipeline_graph
from system.LLMGenerator import LLMGenerator

# --- End Imports ---
MODEL_PATH = "models/risk_model.pth"
TOKENIZER_PATH = "models/tokenizer.joblib"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 4. Global Agent Initialization ---
# Setup logger before initialization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    retrieval_agent = RetrievalAgent()
    predictive_agent = PredictiveAgent()
    input_manager = InputManager()
    logger.info("All MAS agents (RetrievalAgent, PredictiveAgent, InputManager) initialized.")

except Exception as e:
    logger.error(f"Critical error during agent initialization. Error: {e}")
    raise


# --- 5. System Pipeline Function (The actual execution path) ---

async def run_pipeline_simple(user_input: str) -> Dict[str, Any]:
    """
    Runs a simplified, linear chain representing the MAS flow.
    This function demonstrates the execution *without* using the LangGraph object.
    """
    state = {
        "current_input": user_input,
        "validated_input": None,
        "predictive_result": {},
        "error": None,
        # Mock RAG outputs for the simple execution path
        "rag_answer": "",
        "retrieved_context": "",
    }
    logger.info(f"--- Starting MAS Pipeline for: '{user_input}' ---")

    # 1. InputManager (Sync)
    validated_query = input_manager.validate_input(user_input)
    if validated_query.startswith("Error:"):
        state["error"] = validated_query
        return state

    state["validated_input"] = validated_query
    state["current_input"] = validated_query
    logger.info("Step 1: Input Validation Complete.")

    # 2. RetrievalAgent (Sync Mock)
    # The actual retrieval logic populates rag_answer and retrieved_context
    state = retrieval_agent.run(state)
    logger.info("Step 2: RetrievalAgent (RAG) Complete.")

    # 3. PredictiveAgent (Async Await)
    risk_assessment = await predictive_agent.predict_risk(
        rag_answer=state.get("rag_answer", ""),
        retrieved_context=state.get("retrieved_context", "")
    )

    state["predictive_result"] = {
        "risk": risk_assessment.get("risk"),
        "score": risk_assessment.get("score"),
        "method": risk_assessment.get("method"),
        "plan": risk_assessment.get("plan"),
    }

    logger.info("Step 3: PredictiveAgent Complete.")
    return state


# --- Main Execution (Async Handler) ---

async def main():
    """Wrapper function to run the test cases asynchronously."""


    print("\n------------------------------------------------")
    print("--- Test Case 1: Valid High Risk Query (ML Score: 0.793) ---")
    query_1 = "Patient reports extreme thirst and mild blurred vision in the morning. They also have a history of high blood sugar."

    final_state_1 = await run_pipeline_simple(query_1)
    print("\n--- Final State 1 Output ---")
    pprint(final_state_1)

    print("\n------------------------------------------------")
    print("--- Test Case 2: Valid Moderate Risk Query (ML Score: 0.450) ---")
    query_3 = "Patient reports general, mild fatigue for two days but no other symptoms. They feel fine otherwise."

    final_state_3 = await run_pipeline_simple(query_3)
    print("\n--- Final State 2 Output ---")
    pprint(final_state_3)

    print("\n------------------------------------------------")
    print("--- Test Case 3: Invalid (Too Short) Query ---")
    query_2 = "tired"

    final_state_2 = await run_pipeline_simple(query_2)
    print("\n--- Final State 3 Output ---")
    pprint(final_state_2)

    # Build and show the graph structure (not for execution, but for visualization)
    graph_app = build_pipeline_graph()
    logger.info(f"LangGraph compiled with {len(graph_app.nodes)} nodes: {list(graph_app.nodes.keys())}")
    # In a real environment, you might display graph_app.get_graph().draw_png()
    # 4. Mermaid Diagram Output
    print("\n================ MERMAID DIAGRAM =============")

    mermaid_base = graph_app.get_graph().draw_mermaid()

    # Custom stylesheet (your provided colors)
    # Custom styles to visually differentiate nodes
    custom_styles = """
    classDef default fill:#D4EDDA, stroke:#4CAF50, color:#155724;
    classDef first fill:#CCE5FF, stroke:#007BFF, color:#004085;
    classDef last fill:#F8D7DA, stroke:#DC3545, color:#721C24;
    """
    print(mermaid_base + custom_styles)


if __name__ == "__main__":
    try:
        # Crucial: Use asyncio.run to execute the top-level async function
        # The linear run_pipeline_simple is used for execution, not the LangGraph object
        asyncio.run(main())
    except RuntimeError as e:
        print(f"Execution finished or encountered a runtime error: {e}")
# try:
#     # We no longer need to instantiate LLMGenerator globally.
#     # predictive_agent handles its own internal LLM instantiation.
#     retrieval_agent = RetrievalAgent()
#     predictive_agent = PredictiveAgent()
#     input_manager = InputManager()
#     logger.info("All MAS agents (RetrievalAgent, PredictiveAgent, InputManager) initialized.")
#
# except Exception as e:
#     logger.error(f"Critical error during agent initialization. Error: {e}")
#     raise
#
#
# # --- 5. System Pipeline Function ---
#
# def run_pipeline_simple(user_input: str) -> Dict[str, Any]:
#     """
#     Runs a simplified chain representing the MAS flow:
#     InputManager -> RetrievalAgent -> PredictiveAgent.
#     """
#
#     # 1. Initial State Setup
#     state = {
#         "current_input": user_input,
#         "validated_input": None,
#         "predictive_result": {},
#         "error": None,
#     }
#     logger.info(f"--- Starting MAS Pipeline for: '{user_input}' ---")
#
#     # 2. Step 1: Execute InputManager (Validation)
#     validated_query = input_manager.validate_input(user_input)
#
#     if validated_query.startswith("Error:"):
#         state["error"] = validated_query
#         logger.error(f"Input validation failed: {validated_query}")
#         return state
#
#     state["validated_input"] = validated_query
#     state["current_input"] = validated_query
#     logger.info("Step 1: Input Validation Complete.")
#
#     # 3. Step 2: Execute RetrievalAgent (RAG Subsystem)
#     rag_results = retrieval_agent.run(state)
#     state.update(rag_results)
#     logger.info("Step 2: RetrievalAgent (RAG) Complete.")
#
#     # Extract RAG outputs needed for the next step
#     rag_answer = state.get("rag_answer", "")
#     retrieved_context = state.get("retrieved_context", "")
#
#     # 4. Step 3: Execute PredictiveAgent Logic
#     risk_assessment = predictive_agent.predict_risk(
#         rag_answer=rag_answer,
#         retrieved_context=retrieved_context
#     )
#
#     # 4b. Combine all predictive results (The plan is already in risk_assessment)
#     combined_result = {
#         "risk": risk_assessment.get("risk"),
#         "score": risk_assessment.get("score"),
#         "method": risk_assessment.get("method"),
#         # The plan is generated by the LLM inside predict_risk
#         "plan": risk_assessment.get("plan"),
#     }
#
#     # Update state with the final prediction
#     state["predictive_result"] = combined_result
#     logger.info("Step 3: PredictiveAgent Complete.")
#
#     return state
#
#
# if __name__ == "__main__":
#     print("\n------------------------------------------------")
#     print("--- Test Case 1: Valid High Risk Query (ML > 0.70) ---")
#     query_1 = "Patient reports extreme thirst and mild blurred vision in the morning. They also have a history of high blood sugar."
#     final_state_1 = run_pipeline_simple(query_1)
#     pprint(final_state_1)
#
#     print("\n------------------------------------------------")
#     print("--- Test Case 2: Valid Moderate Risk Query (ML > 0.40) ---")
#     query_3 = "Patient reports general, mild fatigue for two days but no other symptoms. They feel fine otherwise."
#     final_state_3 = run_pipeline_simple(query_3)
#     pprint(final_state_3)
#
#     print("\n------------------------------------------------")
#     print("--- Test Case 3: Invalid (Too Short) Query ---")
#     query_2 = "tired"
#     final_state_2 = run_pipeline_simple(query_2)
#     pprint(final_state_2)

# --- 1. Global Agent Initialization ---
# # Initialize the LLM needed for the Predictive Agent's reasoning fallback
# try:
#     # 1. Initialize the LLM Generator
#     llm_generator = LLMGenerator()
#
#     # 2. Initialize the primary agents and utilities
#     retrieval_agent = RetrievalAgent()
#     predictive_agent = PredictiveAgent()
#     input_manager = InputManager()
#     logger.info("All MAS agents (RetrievalAgent, PredictiveAgent, InputManager) initialized.")
#
# except Exception as e:
#     logger.error(f"Critical error during agent initialization. The script will now stop. Error: {e}")
#     # Allowing the exception to propagate will cause the script to crash,
#     # which is desirable behavior for failed test environment setup.
#     raise
#
#
# # --- End Initialization ---
#
#
# def run_pipeline_simple(user_input: str) -> Dict[str, Any]:
#     """
#     Runs a simplified chain representing the MAS flow:
#     InputManager -> RetrievalAgent -> PredictiveAgent.
#     """
#
#     # 1. Initial State Setup
#     state = {
#         "current_input": user_input,
#         "validated_input": None,
#         "predictive_result": {},
#         "error": None,  # Field to store any validation error
#     }
#     logger.info(f"--- Starting MAS Pipeline for: '{user_input}' ---")
#
#     # 2. Step 1: Execute InputManager (Validation)
#     validated_query = input_manager.validate_input(user_input)
#
#     if validated_query.startswith("Error:"):
#         state["error"] = validated_query
#         logger.error(f"Input validation failed: {validated_query}")
#         return state  # Halt pipeline and return state with error
#
#     state["validated_input"] = validated_query
#     # Use the validated query for the next steps
#     state["current_input"] = validated_query
#     logger.info("Step 1: Input Validation Complete.")
#
#     # 3. Step 2: Execute RetrievalAgent (RAG Subsystem)
#     # The RetrievalAgent handles the full RAG process and updates the state.
#     rag_results = retrieval_agent.run(state)
#     state.update(rag_results)
#     logger.info("Step 2: RetrievalAgent (RAG) Complete.")
#
#     # Extract RAG outputs needed for the next step
#     rag_answer = state.get("rag_answer", "")
#     retrieved_context = state.get("retrieved_context", "")
#
#     # 4. Step 3: Execute PredictiveAgent Logic
#
#     # 4a. Get the risk prediction (ML + LLM Fallback)
#     risk_assessment = predictive_agent.predict_risk(
#         rag_answer=rag_answer,
#         retrieved_context=retrieved_context
#     )
#
#     # 4b. Generate a human-readable plan
#     risk = risk_assessment.get("risk", "low")
#     plan_result = predictive_agent.generate_plan(risk)
#
#     # 4c. Combine all predictive results
#     combined_result = {
#         "risk": risk,
#         "score": risk_assessment.get("score"),
#         "method": risk_assessment.get("method"),
#         "plan": plan_result.get("plan"),
#     }
#
#     # Update state with the final prediction
#     state["predictive_result"] = combined_result
#     logger.info("Step 3: PredictiveAgent Complete.")
#
#     return state
#
#
# if __name__ == "__main__":
#     print("\n------------------------------------------------")
#     print("--- Test Case 1: Valid High Risk Query ---")
#     query_1 = "Patient reports extreme thirst and mild blurred vision in the morning. They also have a history of high blood sugar."
#     final_state_1 = run_pipeline_simple(query_1)
#     pprint(final_state_1)
#
#     print("\n------------------------------------------------")
#     print("--- Test Case 2: Invalid (Too Short) Query ---")
#     query_2 = "tired"
#     final_state_2 = run_pipeline_simple(query_2)
#     pprint(final_state_2)
# from pipeline.nodes.retrieval_node import retrieval_node
# from pipeline.nodes.predictive_node import predictive_node
#
# def run_pipeline_simple(user_input: str):
#     state = {
#         "current_input": user_input,
#         "validated_input": {"validated": user_input},  # replace with InputManager
#         "retrieved_context": None,
#         "rag_answer": None,
#         "rag_summary": None,
#     }
#
#     # Run retrieval (RAG pipeline)
#     rag_updates = retrieval_node(state)
#     state.update(rag_updates)
#
#     # Run prediction
#     pred_updates = predictive_node(state)
#     state.update(pred_updates)
#
#     return state
#
# if __name__ == "__main__":
#     s = run_pipeline_simple("Patient has high blood sugar and fatigue")
#     from pprint import pprint
#     pprint(s)
