from interfaces.system.ICheckpointer import ICheckpointer
from system.AgentState import AgentState

class Checkpointer(ICheckpointer):
    def __init__(self):
        self.saved_states = []

    def save_state(self, agent_state: AgentState):
        # Save a copy of the agent state
        self.saved_states.append(agent_state.get_state())

    def load_state(self, index: int) -> dict:
        if 0 <= index < len(self.saved_states):
            return self.saved_states[index]
        return {}
