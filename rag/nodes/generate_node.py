import logging

from interfaces.rag.IGenerator import IGenerator
from rag.state.RagAgentState import RagAgentState


logger = logging.getLogger(__name__)

def generate_node(state):
    # Convert dict to RagAgentState if necessary
    if isinstance(state, dict):
        rag_state = RagAgentState()
        rag_state.generator = state.get("generator")
        rag_state.retrieved_docs = state.get("retrieved_docs", [])
        rag_state.query = state.get("current_input", "")
        state = rag_state

    generator: IGenerator = state.generator
    if not generator or not state.retrieved_docs:
        logger.warning("[generate_node] Generator or retrieved_docs not available")
        state.answer = ""
        return state

    doc_texts = [doc.page_content for doc in state.retrieved_docs]
    state.answer = generator.generate(state.query, doc_texts)
    logger.info("[generate_node] Answer generated")
    return state


# def generate_node(state: RagAgentState) -> RagAgentState:
#     generator: IGenerator = state.generator
#     if not generator or not state.retrieved_docs:
#         logger.warning("[generate_node] Generator or retrieved_docs not available")
#         state.answer = ""
#         return state
#
#     # Extract text content from Document objects
#     doc_texts = [doc.page_content for doc in state.retrieved_docs]
#
#     # Generate answer using the generator interface
#     state.answer = generator.generate(state.query, doc_texts)
#     logger.info("[generate_node] Answer generated")
#     return state



# import logging
# from typing import List
#
# from interfaces.rag.IGenerator import IGenerator
# from rag.state.RagAgentState import RagAgentState
#
#
# logger = logging.getLogger(__name__)
#
# def generate_node(state: RagAgentState) -> RagAgentState:
#     generator: IGenerator = state.generator
#     retrieved_docs: List[str] = [doc.page_content for doc in state.retrieved_docs]
#     query: str = state.query
#
#     if not generator or not retrieved_docs:
#         logger.warning("[generate_node] Generator or retrieved_docs not available")
#         state.answer = ""
#         return state
#
#     # Use the interface method
#     state.answer = generator.generate(query, retrieved_docs)
#     logger.info("[generate_node] Answer generated")
#     return state
#







# from typing import TypedDict
# from rag.generator.LLMGenerator import LLMGenerator  # Tool is in the parent directory of this module
#
#
# # Define the expected return structure for the generation node
# class GenerationUpdate(TypedDict):
#     final_response: str
#
#
# def generation_node(agent_state: dict) -> GenerationUpdate:
#     """
#     ORCHESTRATOR: Synthesizes the final response by calling the LLM Generator.
#     This node extracts query and context from AgentState and returns the result to AgentState.
#     """
#
#     print("[Generation Node]: Synthesizing final response...")
#
#     # 1. Extract necessary inputs from state
#     user_query = agent_state.get("current_input", "No query provided.")
#     context = agent_state.get("retrieved_context", "No context found.")
#
#     if not context or "No context found" in context:
#         print("[Generation Node]: Error - Missing context for generation.")
#         return {"final_response": "Error: Cannot generate response without retrieved context."}
#
#     # 2. Define the state prompt for the LLM
#     system_prompt = (
#         "You are a compassionate and expert clinical assistant. "
#         "Your goal is to provide a clear, empathetic, and strictly evidence-based "
#         "summary of the patient's potential condition and next steps, "
#         "based only on the RETRIEVED CONTEXT provided."
#     )
#
#     # 3. Initialize the LLM Generator pure tool
#     generator = LLMGenerator()
#
#     # 4. Call the pure tool method (no AgentState passed to service)
#     final_answer = generator.generate_response(
#         user_query=user_query,
#         context=context,
#         system_prompt=system_prompt
#     )
#
#     print(f"[Generation Node]: Synthesis complete. Response received.")
#
#     # 5. Return the final answer as a state update
#     return {
#         "final_response": final_answer
#     }