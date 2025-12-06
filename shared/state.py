from interfaces.system.IAgentState import IAgentState

# state/AgentState.py (Refactored)
from typing import TypedDict, Optional, Any, List


# Define the state schema using TypedDict
# This defines the data schema (structure) for the graph state.
from typing import Optional, List, Dict
from typing import Optional, List, TypedDict


# ---------------------------
# Define the AgentState schema
# ---------------------------
# class AgentState(TypedDict):
#     current_input: Optional[str]
#     validated_input: Optional[dict]
#     augmented_input: Optional[dict]
#     embedded_query: Optional[List[float]]
#     retrieved_context: Optional[str]
#     rag_answer: Optional[str]
#     rag_summary: Optional[str]
#     metadata: dict

class AgentState(TypedDict):
    current_input: Optional[str]
    validated_input: Optional[dict]
    retrieved_context: Optional[str]
    rag_answer: Optional[str]
    rag_summary: Optional[str]
    metadata: dict




