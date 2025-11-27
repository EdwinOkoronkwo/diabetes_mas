# run_pipeline.py

# run_pipeline.py

from pipeline.build_graph import build_pipeline_graph
import sys

from pipeline.nodes.predictive_node import predictive_node
from pipeline.nodes.retrieval_node import retrieval_node
from system.AgentState import AgentState

def run_pipeline(user_input: str) -> AgentState:

    # Initial state
    state: AgentState = {
        "current_input": user_input,
        "validated_input": {"validated": user_input},   # TEMP: replace by real validation agent
        "retrieved_context": None,
        "rag_answer": None,
        "rag_summary": None,
        "predicted_risk": None,
        "recommended_plan": None,
        "metadata": {},
    }

    # 1. RETRIEVAL (RAG pipeline)
    rag_updates = retrieval_node(state)
    state.update(rag_updates)

    # 2. PREDICTION
    pred_updates = predictive_node(state)
    state.update(pred_updates)

    return state


if __name__ == "__main__":
    result = run_pipeline("Patient has high blood sugar and fatigue")

    print("\n=== FINAL OUTPUT ===")
    for k, v in result.items():
        print(f"{k}: {v if len(str(v)) < 200 else str(v)[:200] + '...'}")



# def main():
#     print("=== Building MAS + RAG Pipeline ===")
#
#     # 1. Build LangGraph application
#     try:
#         app = build_pipeline_graph()
#     except Exception as e:
#         print(f"\n❌ ERROR: Failed to compile LangGraph.\nDetails: {e}")
#         sys.exit(1)
#
#     # 2. Initial AgentState for execution
#     initial_state_dict: AgentState = {
#         "current_input": "Patient has high blood sugar and fatigue",
#         "validated_input": None,
#         "embedded_query": None,
#         "retrieved_context": None,
#         "rag_answer": None,
#         "rag_summary": None,
#         "metadata": {},
#     }
#
#     print("\n=== Running LangGraph Pipeline ===")
#
#     # FIX: Add config with thread_id
#     result_state = app.invoke(initial_state_dict, config={"configurable": {"thread_id": "pipeline_run_1"}})
#
#     # 3. Output results
#     print("\n================ FINAL STATE ================")
#
#     for key, val in result_state.items():
#         if isinstance(val, list) and key == "embedded_query":
#             print(f"{key}: [Vector of size {len(val)}]")
#         elif isinstance(val, str) and len(val) > 120:
#             print(f"{key}: {val[:120]}... [Truncated]")
#         else:
#             print(f"{key}: {val}")
#
#     print("\n================ RAG OUTPUT ==================")
#     print("\n--- RETRIEVED CONTEXT ---")
#     print(result_state.get("retrieved_context", "No context found."))
#
#     print("\n--- RAG ANSWER (Mistral Synthesis) ---")
#     print(result_state.get("rag_answer", "No answer."))
#
    # # 4. Mermaid Diagram Output
    # print("\n================ MERMAID DIAGRAM =============")
    #
    # mermaid_str = app.get_graph().draw_mermaid()
    #
    # custom_styles = """
    # classDef default fill:#D4EDDA, stroke:#4CAF50, color:#155724;
    # classDef first fill:#CCE5FF, stroke:#007BFF, color:#004085;
    # classDef last fill:#F8D7DA, stroke:#DC3545, color:#721C24;
    # """
    #
    # print(mermaid_str + custom_styles)


# if __name__ == "__main__":
#     main()

# from pipeline.build_graph import build_pipeline_graph
# import sys
#
# from pipeline.build_graph import build_pipeline_graph
# import sys
# # Import the required AgentState schema to ensure correct type checking and completeness
# from system.AgentState import AgentState
#
#
# def main():
#     print("=== Building MAS + RAG Pipeline ===")
#
#     # 1. Build LangGraph application
#     try:
#         app = build_pipeline_graph()
#     except Exception as e:
#         print(f"\n❌ ERROR: Failed to compile LangGraph.\nDetails: {e}")
#         # Exit if compilation fails
#         sys.exit(1)
#
#     # 2. Initial AgentState for execution
#     # Ensure all keys from the AgentState TypedDict are present and match the schema.
#     initial_state_dict: AgentState = {
#         "current_input": "Patient has high blood sugar and fatigue",
#         "validated_input": None,
#         # Removed 'augmented_input' as it is not in the AgentState schema.
#         "embedded_query": None,
#         "retrieved_context": None,
#         "rag_answer": None,
#         "rag_summary": None,
#         "metadata": {},
#     }
#
#     print("\n=== Running LangGraph Pipeline ===")
#
#     # Run the pipeline synchronously
#     result_state = app.invoke(initial_state_dict)
#
#     # 3. Output results
#     print("\n================ FINAL STATE ================")
#
#     # Print the full state with special handling for large vectors and strings
#     for key, val in result_state.items():
#         if isinstance(val, list) and key == "embedded_query":
#             print(f"{key}: [Vector of size {len(val)}]")
#         elif isinstance(val, str) and len(val) > 120:
#             print(f"{key}: {val[:120]}... [Truncated]")
#         else:
#             print(f"{key}: {val}")
#
#     print("\n================ RAG OUTPUT ==================")
#     print("\n--- RETRIEVED CONTEXT ---")
#     print(result_state.get("retrieved_context", "No context found."))
#
#     print("\n--- RAG ANSWER (Mistral Synthesis) ---")
#     print(result_state.get("rag_answer", "No answer."))
#
#     # 4. Mermaid Diagram Output
#     print("\n================ MERMAID DIAGRAM =============")
#
#     mermaid_str = app.get_graph().draw_mermaid()
#
#     custom_styles = """
#     classDef default fill:#D4EDDA, stroke:#4CAF50, color:#155724;
#     classDef first fill:#CCE5FF, stroke:#007BFF, color:#004085;
#     classDef last fill:#F8D7DA, stroke:#DC3545, color:#721C24;
#     """
#
#     # We are using a full 4-node RAG structure:
#     # START --> InputManager --> Embedding --> Retrieval --> Generation --> END
#     #
#
#     print(mermaid_str + custom_styles)
#
#
# if __name__ == "__main__":
#     main()

# def main():
#     print("=== Building MAS + RAG Pipeline ===")
#
#     # 1. Build LangGraph application
#     try:
#         app = build_pipeline_graph()
#     except Exception as e:
#         print(f"\n❌ ERROR: Failed to compile LangGraph.\nDetails: {e}")
#         sys.exit(1)
#
#     # 2. Initial AgentState for execution
#     initial_state_dict = {
#         "current_input": "Patient has high blood sugar and fatigue",
#         "validated_input": None,
#         "augmented_input": None,
#         "retrieved_context": None,
#         "rag_answer": None,
#         "rag_summary": None,
#         "metadata": {},
#         "embedded_query": None,
#     }
#
#     print("\n=== Running LangGraph Pipeline ===")
#     result_state = app.invoke(initial_state_dict)
#
#     # 3. Output results
#     print("\n=== FINAL STATE ===")
#     for key, val in result_state.items():
#         print(f"{key}: {val}")
#
#     print("\n=== RETRIEVED CONTEXT ===")
#     print(result_state.get("retrieved_context", "No context found."))
#
#     print("\n=== RAG ANSWER ===")
#     print(result_state.get("rag_answer", "No answer."))
#
#     print("\n=== RAG SUMMARY ===")
#     print(result_state.get("rag_summary", "No summary."))
#
#     # 4. Mermaid Diagram Output
#     print("\n=== MERMAID DIAGRAM ===")
#
#     mermaid_str = app.get_graph().draw_mermaid()
#
#     custom_styles = """
#     classDef default fill:#D4EDDA, stroke:#4CAF50, color:#155724;
#     classDef first fill:#CCE5FF, stroke:#007BFF, color:#004085;
#     classDef last fill:#F8D7DA, stroke:#DC3545, color:#721C24;
#     """
#
#     print(mermaid_str + custom_styles)
#
#
# if __name__ == "__main__":
#     main()

