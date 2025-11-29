# diabetes_mas/pipeline/nodes/io_manager_node.py

from typing import TypedDict
# Assuming these are necessary for your logic and state definition
from agents.IOManager import IOManager
from system.AgentState import AgentState


# ============================================
# io_manager_node.py
# ============================================

def io_manager_node(agent_state: dict) -> dict:
    """
    Validates the input and stores it in validated_input.
    """
    manager = IOManager()
    current_input = agent_state.get("current_input", "")

    # Validate the input
    validated = manager.validate_input(current_input)
    
    # Update state
    agent_state["validated_input"] = validated
    
    return agent_state
    validated = manager.validate_input(current_input)

    print(f"[InputManager]: Input validated: '{validated}'")

    # Return the validated input directly to the state
    return {
        "validated_input": validated
    }

# def input_manager_node(agent_state: AgentState) -> AgentState:
#     user_input = agent_state["current_input"]
#     manager = InputManager()
#     validated = manager.validate_input(user_input)
#     return {"validated_input": validated}  # matches schema

# class InputUpdate(TypedDict):
#     validated_input: str
#
#
# def input_manager_node(agent_state: AgentState) -> InputUpdate:
#     """
#     ORCHESTRATOR: Validates the initial user input using the InputManager service.
#     """
#     print("[InputManager Node]: Starting input validation...")
#
#     user_input = agent_state["current_input"]
#     manager = InputManager()
#     validated = manager.validate_input(user_input)
#
#     print("[InputManager Node]: Validation complete.")
#
#     return {"validated_input": validated}