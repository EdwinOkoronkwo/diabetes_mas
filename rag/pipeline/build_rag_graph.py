from langgraph.graph import StateGraph

from rag.nodes.embed_node import embed_node
from rag.nodes.retrieval_node import retrieval_node
from rag.nodes.generate_node import generate_node
from rag.nodes.summarize_node import summarize_node

def build_rag_graph():

    graph = StateGraph()

    graph.add_node("embed", embed_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generate_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge("embed", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "summarize")

    graph.set_entry_point("embed")
    graph.set_finish_point("summarize")

    return graph.compile()
