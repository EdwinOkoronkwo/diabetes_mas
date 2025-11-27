from typing import Dict

from interfaces.system.AbstractLLMInterpreter import AbstractLLMInterpreter


class LLMInterpreter(AbstractLLMInterpreter):
    """
    Concrete implementation of AbstractLLMInterpreter.
    Simulates the structured call to the Gemini API.
    """

    def interpret(self, symptom_text: str, ml_score: float) -> Dict[str, str]:
        """
        Interprets the ML score and symptoms using an LLM to generate
        a dynamic risk level and action plan in a structured JSON format.
        (NOTE: This is a simulation, as the actual API call would require
        async handling and network I/O.)
        """
        # --- START: LLM API Call Simulation ---

        # In a real system, the actual API call logic for structured output would go here.

        if ml_score >= 0.70:
            return {
                "riskLevel": "High",
                "plan": "Based on the severe symptoms and high objective score, advise immediate and urgent clinical assessment, including comprehensive blood work (FPG/A1c)."
            }
        elif ml_score >= 0.40:
            return {
                "riskLevel": "Moderate",
                "plan": "Recommend a prompt consultation with a primary care physician within one week for diagnostic testing and discussion of preventative lifestyle adjustments."
            }
        else:
            return {
                "riskLevel": "Low",
                "plan": "Maintain current healthy lifestyle, focus on diet and exercise, and re-check symptoms if they worsen or persist over two weeks."
            }
        # --- END: LLM API Call Simulation ---
