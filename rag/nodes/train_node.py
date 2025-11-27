def train_node(state):
    trainer = state["trainer"]
    trainer.fine_tune()
    return state
