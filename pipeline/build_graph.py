# diabetes_mas/pipeline/build_graph.py
# from langgraph.graph import StateGraph, START, END
from langgraph.graph import StateGraph, END

from pipeline.nodes.predictive_node import predictive_node
from system.AgentState import AgentState
from pipeline.nodes.input_manager_node import input_manager_node
from pipeline.nodes.retrieval_node import retrieval_node


def build_pipeline_graph():
    """Defines the structure of the Multi-Agent System using LangGraph."""
    # Modern syntax (LangGraph 0.2+)
    graph_builder = StateGraph(AgentState)

    # Define Node Names
    input_node_name = "InputManagerNode"
    retrieval_node_name = "RetrievalAgentNode"
    predictive_node_name = "PredictiveAgentNode" # Added the predictive node

    # Add Nodes
    graph_builder.add_node(input_node_name, input_manager_node)
    graph_builder.add_node(retrieval_node_name, retrieval_node)
    graph_builder.add_node(predictive_node_name, predictive_node) # Added the predictive node

    # Set Entry Point
    graph_builder.set_entry_point(input_node_name)

    # Add Edges (Linear Flow)
    graph_builder.add_edge(input_node_name, retrieval_node_name)
    graph_builder.add_edge(retrieval_node_name, predictive_node_name)
    graph_builder.add_edge(predictive_node_name, END)

    app = graph_builder.compile()
    return app


