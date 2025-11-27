from langgraph.graph import StateGraph, START, END
from state.AgentState import AgentState
from pipeline.nodes.input_manager_node import input_manager_node
from pipeline.nodes.retrieval_node import retrieval_node
import sys


# This file runs a focused pipeline testing only the RAG components: Input -> Retrieval.
# The Input Manager node has been removed to treat the RAG pipeline as standalone.

def build_rag_test_graph():
    """
    Defines and compiles the graph for the RAG test pipeline (Input -> Retrieval).

    The graph now runs directly from START to retrieval_node.
    """

    # 1. Create the graph builder with the State Schema
    graph_builder = StateGraph(state_schema=AgentState)

    # 2. Define Nodes
    # The input_manager_node is no longer needed in this standalone RAG test
    retrieval_name = retrieval_node.__name__

    # Add Nodes
    graph_builder.add_node(retrieval_name, retrieval_node)

    # 3. Define Sequential Edges
    # The START connects directly to the retrieval node
    graph_builder.add_edge(START, retrieval_name)
    graph_builder.add_edge(retrieval_name, END)

    # 4. Compile and return the executable application
    app = graph_builder.compile()
    return app


def main():
    # 1. Build and compile the executable graph
    try:
        app = build_rag_test_graph()
    except Exception as e:
        print(f"ERROR: Failed to compile the RAG test graph. Details: {e}")
        sys.exit(1)

    # --- RAG STANDALONE ADJUSTMENT ---
    # Since we removed the input_manager_node, we must manually structure the
    # initial state to mimic its output, ensuring the retrieval_node has its input.
    MOCK_INPUT = "Patient has high blood sugar and fatigue"

    ## 2. Define the Initial State
    initial_state_dict = {
        "current_input": MOCK_INPUT,
        # Mock the output of the input_manager_node so retrieval_node works as expected
        "current_output": {
            "validated_input": {
                "validated": MOCK_INPUT
            }
        },
        "metadata": {},
        "retrieved_context": None,
    }
    # ---------------------------------

    # 3. Run the compiled graph using .invoke()
    print("--- Running Focused RAG Pipeline: Retrieval Standalone ---")
    print(f"--- Mock Input: '{MOCK_INPUT}' ---")
    result_state = app.invoke(initial_state_dict)

    # 4. Print the results
    print("\n--- Graph Execution Result ---")
    print(f"Final State (Full): {result_state}")

    # Access the retrieved context
    retrieved_data = result_state.get("retrieved_context", "Context not found.")
    print(f"\nRetrieved Context:\n{'-' * 20}\n{retrieved_data}\n{'-' * 20}")

    # 5. Print Mermaid diagram
    print("\n--- Mermaid Diagram ---")
    mermaid_str = app.get_graph().draw_mermaid()

    custom_styles = """
    classDef default fill:#D4EDDA, stroke:#4CAF50, color:#155724;
    classDef first fill:#CCE5FF, stroke:#007BFF, color:#004085;
    classDef last fill:#F8D7DA, stroke:#DC3545, color:#721C24;
    """

    print(mermaid_str + custom_styles)


if __name__ == "__main__":
    main()