from interfaces.agents.IKnowledgeBase import IKnowledgeBase

class KnowledgeBase(IKnowledgeBase):
    def __init__(self):
        self.store = {}

    def lookup(self, key: str) -> dict:
        return self.store.get(key, {})

    def store_fact(self, key: str, value: dict):
        self.store[key] = value

    def get_medical_rules(self) -> dict:
        return {"rule1": "example"}
