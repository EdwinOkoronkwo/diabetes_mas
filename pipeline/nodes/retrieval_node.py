# diabetes_mas/pipeline/nodes/retrieval_node.py (Final)

from typing import TypedDict
# Assuming these imports match your structure
from agents.RetrievalAgent import RetrievalAgent
from system.AgentState import AgentState


# Define the expected return structure
class RetrievalUpdate(TypedDict):
    retrieved_context: str


# ============================================
# retrieval_node.py
# ============================================

from agents.RetrievalAgent import RetrievalAgent
from rag.pipeline.run_rag_pipeline import run_rag_pipeline


def retrieval_node(agent_state: dict) -> dict:
    """
    Retrieves context using the RetrievalAgent (which internally runs RAG).
    """
    validated_input = agent_state.get("validated_input")

    # Extract the user query
    if isinstance(validated_input, dict):
        user_query = validated_input.get("validated", "")
    elif isinstance(validated_input, str):
        user_query = validated_input
    else:
        user_query = ""

    if not user_query:
        return {
            "retrieved_context": "ERROR: No valid query extracted.",
            "rag_answer": "",
            "rag_summary": ""
        }

    print(f"[RetrievalAgent]: Processing query: '{user_query}'")

    # Create retrieval agent (mock LLM optional)
    retrieval_agent = RetrievalAgent(use_mock_llm=False)

    # Create temporary internal RAG state
    temp_state = {"current_input": user_query}

    # Run RAG pipeline
    result_state = retrieval_agent.run(temp_state)

    print(f"[DEBUG] RAG keys: {list(result_state.keys())}")
    print(f"[DEBUG] context sample: {str(result_state.get('retrieved_context'))[:100]}...")
    print(f"[DEBUG] answer sample: {str(result_state.get('rag_answer'))[:100]}...")

    # Push back into MAS agent state
    return {
        "retrieved_context": result_state.get("retrieved_context", ""),
        "rag_answer": result_state.get("rag_answer", ""),
        "rag_summary": result_state.get("rag_summary", "")
    }

# def retrieval_node(agent_state: dict) -> dict:
#     """
#     Retrieves context based on the validated input.
#     """
#     # Get validated_input directly from state
#     validated_input = agent_state.get("validated_input")
#
#     # Handle both dict and string formats
#     if isinstance(validated_input, dict):
#         user_query = validated_input.get("validated", "")
#     elif isinstance(validated_input, str):
#         user_query = validated_input
#     else:
#         user_query = ""
#
#     if not user_query:
#         return {"retrieved_context": "ERROR: No valid query extracted."}
#
#     print(f"[RetrievalAgent]: Retrieving context for query: '{user_query}'")
#
#     retriever_agent = RetrievalAgent()
#     context_string = retriever_agent.retrieve_context(user_query)
#
#     return {
#         "retrieved_context": context_string
#     }

# def retrieval_node(agent_state: AgentState) -> AgentState:
#     validated_input = agent_state.get("validated_input", {})
#     query = validated_input.get("validated", "")
#     retriever_agent = RetrievalAgent()
#     context = retriever_agent.retrieve_context(query)
#     return {"retrieved_context": context}



    print(f"[Retrieval Node Debug]: Extracted Query: '{user_query}'")

    # Final check to ensure we have a query string
    if not user_query:
        print("[Retrieval Node Debug]: Error - Query extraction failed.")
        return {"retrieved_context": "ERROR: Could not extract valid string query for retrieval."}

    try:
        # 2. Initialize the RetrievalAgent (uses MockRetriever by default for now)
        retriever_agent = RetrievalAgent()

        # 3. Call the core retrieval method
        # This calls the mock retriever's logic.
        context_string = retriever_agent.retrieve_context(query=user_query)

        print(f"[Retrieval Node Debug]: Context received (length): {len(context_string)}")

        # 4. Return the context as a state update
        return {
            "retrieved_context": context_string
        }

    except Exception as e:
        # Catch any unexpected errors during agent execution
        print(f"[Retrieval Node ERROR]: An unexpected error occurred during retrieval: {e}")
        return {"retrieved_context": f"CRASH ERROR: {e}. Check RetrievalAgent or MockRetriever setup."}