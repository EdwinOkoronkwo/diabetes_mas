from interfaces.agents.IIOManager import IIOManager
from system.AgentState import AgentState


class IOManager(IIOManager):
    """
    A pure service class to handle input validation, initial sanitization, and output formatting.
    """

    def validate_input(self, state: dict) -> dict:
        """Simulates validation and cleaning of the user query."""
        user_input = state.get("current_input", "")
        
        if not isinstance(user_input, str):
             # Handle case where input might not be a string (though it should be)
             user_input = str(user_input)

        if not user_input or len(user_input.strip()) < 5:
            state["error"] = "Error: Query too short."
            return state

        validated_input = user_input.strip()
        print(f"[IOManager]: Input validated: '{validated_input[:40]}...'")
        state["validated_input"] = validated_input
        return state

    def augment_input(self, state: dict) -> dict:
        """Simulates input augmentation."""
        # Placeholder implementation
        # In a real system, this might add context or expand queries
        return state

# class InputManager:
#     def validate_input(self, state):
#         user_input = state["current_input"]
#         state["validated_input"] = {"validated": user_input}
#         return state
#
#     def augment_input(self, state):
#         user_input = state["current_input"]
#         state["augmented_input"] = {"augmented": user_input}
#         return state



