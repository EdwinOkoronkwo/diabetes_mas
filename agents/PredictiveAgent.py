# agents/PredictiveAgent.py

import logging
from typing import Dict, Any, Optional
import os

from interfaces.system.AbstractLLMInterpreter import AbstractLLMInterpreter
from interfaces.system.AbstractMLModel import AbstractMLModel
from system.MLModelWrapper import MLModelWrapper
from system.PredictiveLLMGenerator import PredictiveLLMGenerator

# NOTE: The LLMGenerator dependency is removed as per user instruction.

# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ["MISTRAL_API_KEY"] = "ECnNejAzltaN1KG0zWdpNtSU9y7Wsezv"
os.environ["MISTRAL_MODEL"] = "mistral-small-latest"

# --- API Constants ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# --- Configuration and Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the model path relative to the current file's location.
# --- FIX: Using the ABSOLUTE path structure provided by the user ---
# Note: Renamed the constant for clarity from TOKENIZER_PATH to ML_PIPELINE_PATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model.pth")


class PredictiveAgent:
    """The PredictiveAgent orchestrates the ML scoring and the LLM interpretation."""

    def __init__(self, ml_model_path: Optional[str] = None):
        # Instantiate the required LLM Interpreter
        self.llm: AbstractLLMInterpreter = PredictiveLLMGenerator()
        self.ml = MLModelWrapper()
        logger.info("[PredictiveAgent] Agents initialized.")

    async def predict_risk(self, rag_answer: str, retrieved_context: str = "") -> Dict[str, Any]:
        """Hybrid prediction: ML for score, LLM for nuanced interpretation and plan."""

        text = f"Symptoms: {rag_answer.split('.')[0]}.\nFull Context: {rag_answer}\nRetrieved Docs: {retrieved_context}"
        text = text.strip() or "no content"

        # 1. Run ML Model for Objective Score
        prob = float(self.ml.predict_proba([text])[0])
        logger.info(f"[PredictiveAgent] Objective ML Score: {prob:.3f}")

        # 2. Run LLM for Interpretation and Structured Output
        llm_result = await self.llm.interpret(text, prob)

        # Extract risk and plan
        risk = llm_result.get("riskLevel", "unknown").lower()
        plan = llm_result.get("plan", "No plan generated.")

        # --- CRITICAL FIX FOR TYPE ERROR ---
        # Ensure 'plan' is a string before slicing it for logging.
        plan_log_display = str(plan)

        logger.info(f"[PredictiveAgent] LLM Interpretation: {risk} (Plan: {plan_log_display[:40]}...)")
        # -----------------------------------

        return {
            "risk": risk,
            "score": prob,
            "plan": plan,
            "method": "hybrid"
        }

# class PredictiveAgent:
#     """
#     The PredictiveAgent uses composition, relying on the defined interfaces
#     to perform its hybrid risk assessment.
#     """
#
#     # Type hints now use the abstract interfaces
#     def __init__(self, llm_interpreter: AbstractLLMInterpreter, ml_model_path: Optional[str] = ML_PIPELINE_PATH):
#         # Composition: The agent USES the LLM interpreter and the ML model.
#         self.llm: AbstractLLMInterpreter = llm_interpreter
#         self.ml: Optional[AbstractMLModel] = None
#
#         if ml_model_path:
#             try:
#                 # We instantiate the concrete implementation (MLModelWrapper)
#                 # but it conforms to the AbstractMLModel interface.
#                 self.ml = MLModelWrapper()
#                 logger.info("[PredictiveAgent] ML score model loaded successfully.")
#             except Exception as e:
#                 logger.error(f"[PredictiveAgent] Failed to load ML model from {ml_model_path}. Details: {e}")
#
#     def predict_risk(self, rag_answer: str, retrieved_context: str = "") -> Dict[str, Any]:
#         """
#         Hybrid prediction:
#         1. Predicts the objective risk score (0.0 to 1.0) using the fast ML model.
#         2. Uses the LLM to interpret the score and symptoms, classifying the risk
#            and generating the clinical action plan.
#         """
#         # A. Prepare Text Input
#         text = (rag_answer or "") + "\n\n" + (retrieved_context or "")
#         text = text.strip() or "no content"
#
#         if not self.ml:
#             raise ValueError("ML model object is None. Failed to initialize AbstractMLModel.")
#
#         # B. Run ML Model for Objective Score (Fast & Deterministic)
#         try:
#             # Calls the predict_proba method defined in the AbstractMLModel interface
#             prob = float(self.ml.predict_proba([text])[0])
#             logger.info(f"[PredictiveAgent] Objective ML Score: {prob:.3f}")
#         except Exception as e:
#             logger.error(f"Error during ML prediction: {e}")
#             raise RuntimeError("ML prediction step failed.")
#
#         # C. Run LLM for Interpretation and Structured Output (Nuanced & Context-Aware)
#         try:
#             # Calls the interpret method defined in the AbstractLLMInterpreter interface
#             llm_result = self.llm.interpret(text, prob)
#             risk = llm_result.get("riskLevel", "unknown").lower()
#             plan = llm_result.get("plan", "No plan generated.")
#             logger.info(f"[PredictiveAgent] LLM Interpretation: {risk} (Plan: {plan[:30]}...)")
#
#             return {
#                 "risk": risk,
#                 "score": prob,
#                 "plan": plan,
#                 "method": "hybrid"
#             }
#         except Exception as e:
#             logger.error(f"Error during LLM interpretation: {e}")
#
#             # Fallback based on raw ML score thresholds
#             fallback_risk = "low"
#             if prob >= 0.75:
#                 fallback_risk = "high"
#             elif prob >= 0.4:
#                 fallback_risk = "moderate"
#
#             return {
#                 "risk": fallback_risk,
#                 "score": prob,
#                 "plan": "LLM interpretation failed; risk based on raw ML score thresholds only. Further human review needed.",
#                 "method": "hybrid_fallback"
#             }

# class PredictiveAgent:
#     """
#     Determines the diabetes risk level using a trained ML model.
#     Fails immediately if the ML model cannot load or predict correctly.
#     """
#     # CRITICAL FIX: Removed 'llm_generator' from the signature to fix TypeError
#     def __init__(self, ml_model_path: Optional[str] = ML_PIPELINE_PATH):
#         self.ml = None
#
#         if ml_model_path:
#             try:
#                 # MLModelWrapper will now attempt to load the model from the specified path
#                 self.ml = MLModelWrapper()
#                 logger.info("[PredictiveAgent] ML model loaded successfully.")
#             except Exception as e:
#                 # The exception will now propagate from MLModelWrapper if the file is missing/corrupted,
#                 # which causes the 'ValueError: ML model object is None' upstream.
#                 logger.error(f"[PredictiveAgent] Failed to load ML model from {ml_model_path}. Details: {e}")
#                 # We intentionally don't re-raise here so 'self.ml' remains None,
#                 # allowing 'predict_risk' to raise the final 'ValueError'.
#
#
#     def predict_risk(self, rag_answer: str, retrieved_context: str = "") -> Dict[str, Any]:
#         """
#         Predicts the risk level using the ML model only.
#         Fails if self.ml is not properly initialized or lacks 'predict_proba'.
#         Returns {'risk': <low|moderate|high>, 'score': float, 'method': 'ml'}
#         """
#         # Combine RAG outputs into a single text input for prediction
#         text = (rag_answer or "") + "\n\n" + (retrieved_context or "")
#         text = text.strip() or "no content"
#
#         # --- Direct ML Prediction (Crash if fails) ---
#         if not self.ml:
#             # If the __init__ failed and self.ml is None, this will raise
#             raise ValueError("ML model object is None. Failed to initialize MLModelWrapper.")
#
#         # This line will now execute the happy path once the model loads correctly.
#         prob = float(self.ml.predict_proba([text])[0])
#
#         if prob >= 0.75:
#             risk = "high"
#         elif prob >= 0.4:
#             risk = "moderate"
#         else:
#             risk = "low"
#
#         logger.info(f"[PredictiveAgent] ML Risk: {risk} (Score: {prob:.2f})")
#         return {"risk": risk, "score": prob, "method": "ml"}
#
#     def generate_plan(self, risk: str) -> Dict[str, Any]:
#         """Return a human-readable plan based on risk."""
#         if risk == "high":
#             plan = "Advise urgent clinical assessment and blood tests (FPG/A1c), consider referral."
#         elif risk == "moderate":
#             plan = "Recommend prompt GP visit for blood tests and lifestyle changes; follow-up in 4 weeks."
#         else:
#             plan = "Advise lifestyle advice; monitor symptoms and re-check if symptoms worsen."
#         return {"plan": plan}


# import logging
# from typing import Dict, Any, Optional
# from agents.ml_model import MLModelWrapper
# from rag.generator.LLMGenerator import LLMGenerator

# # ---------------------------------------------------------------------
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
#
#
# class PredictiveAgent:
#     def __init__(self, ml_model_path: Optional[str] = "models/risk_model.joblib",
#                  llm_generator: Optional[LLMGenerator] = None):
#         self.ml = None
#
#         # --- ML Model Initialization (CRITICAL: Allowing exceptions to propagate) ---
#         if ml_model_path:
#             self.ml = MLModelWrapper(ml_model_path)
#             logger.info("[PredictiveAgent] ML model loaded successfully.")
#
#         self.llm = llm_generator
#
#     def predict_risk(self, rag_answer: str, retrieved_context: str = "") -> Dict[str, Any]:
#         """
#         Predicts the risk level using the ML model first, falling back to LLM reasoning.
#         Returns {'risk': <low|moderate|high>, 'score': float, 'method': <ml|llm|rule>}
#         """
#         # Combine RAG outputs into a single text input for prediction
#         text = (rag_answer or "") + "\n\n" + (retrieved_context or "")
#         text = text.strip() or "no content"
#
#         # Attempt ML model prediction (The "Happy Path")
#         if self.ml:
#             try:
#                 # Prediction probability for the positive class (risk=1)
#                 prob = float(self.ml.predict_proba([text])[0])
#
#                 # Map probability to qualitative risk
#                 if prob >= 0.75:
#                     risk = "high"
#                 elif prob >= 0.4:
#                     risk = "moderate"
#                 else:
#                     risk = "low"
#
#                 logger.info(f"[PredictiveAgent] ML Risk: {risk} (Score: {prob:.2f})")
#                 return {"risk": risk, "score": prob, "method": "ml"}
#             except Exception as e:
#                 # This catches errors *during* prediction (e.g., model corruption, bad input format)
#                 # It now falls back silently after logging, as the LLM/Rule path is preferred over crashing here.
#                 logger.warning(f"[PredictiveAgent] ML prediction failed: {e}. Falling back to LLM.")
#
#         # Fallback to LLM reasoning (Structured Output)
#         if self.llm and hasattr(self.llm, "invoke_structured"):
#             prompt = (
#                 "Given the following retrieved clinical context and model-generated answer, "
#                 "provide a risk assessment (low/moderate/high) and a recommended plan. "
#                 "Return only the structured JSON output."
#                 f"Context:\n{retrieved_context}\n\nAnswer:\n{rag_answer}\n\nAssessment:"
#             )
#
#             response_schema = {
#                 "risk": "string (low, moderate, or high)",
#                 "plan": "string (recommended next step for the patient)",
#                 "confidence": "float (0.0 to 1.0)"
#             }
#
#             try:
#                 llm_json = self.llm.invoke_structured(
#                     messages=[{"role": "user", "content": prompt}],
#                     response_schema=response_schema
#                 )
#
#                 risk = llm_json.get("risk", "low").lower()
#                 score = llm_json.get("confidence", 0.5)
#                 plan = llm_json.get("plan", "No plan generated.")
#
#                 logger.info(f"[PredictiveAgent] LLM Risk: {risk} (Score: {score:.2f})")
#
#                 return {"risk": risk, "score": score, "method": "llm_reasoning", "plan": plan}
#
#             except Exception as e:
#                 logger.warning(f"[PredictiveAgent] LLM parsing failed: {e}. Falling back to simple rules.")
#
#         # Final fallback: simple rules
#         if "blood sugar" in text.lower() or "hba1c" in text.lower() or "fasting" in text.lower():
#             risk = "moderate"
#         elif "fatigue" in text.lower() or "tired" in text.lower():
#             risk = "moderate"
#         else:
#             risk = "low"
#
#         logger.info(f"[PredictiveAgent] Rule-based Risk: {risk}")
#         return {"risk": risk, "score": 0.1, "method": "rule"}
#
#     def generate_plan(self, risk: str) -> Dict[str, Any]:
#         """Return a human-readable plan based on risk."""
#         if risk == "high":
#             plan = "Advise urgent clinical assessment and blood tests (FPG/A1c), consider referral."
#         elif risk == "moderate":
#             plan = "Recommend prompt GP visit for blood tests and lifestyle changes; follow-up in 4 weeks."
#         else:
#             plan = "Advise lifestyle advice; monitor symptoms and re-check if symptoms worsen."
#         return {"plan": plan}
# import logging
# from typing import Dict, Any, List, Optional
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
#
# try:
#     from agents.ml_model import MLModelWrapper
# except Exception:
#     MLModelWrapper = None
#
# try:
#     from rag.generator.LLMGenerator import LLMGenerator
# except Exception:
#     LLMGenerator = None
#
# class PredictiveAgent:
#     def __init__(self, ml_model_path: Optional[str] = "models/risk_model.joblib", llm_generator: Optional[LLMGenerator] = None):
#         self.ml = None
#         if ml_model_path and MLModelWrapper:
#             try:
#                 self.ml = MLModelWrapper(ml_model_path)
#                 logger.info("[PredictiveAgent] ML model loaded")
#             except Exception as e:
#                 logger.warning(f"[PredictiveAgent] Could not load ML model: {e}; will use LLM fallback.")
#
#         self.llm = llm_generator  # instance of LLMGenerator or None
#
#     def predict_risk(self, rag_answer: str, retrieved_context: str = "") -> Dict[str, Any]:
#         """
#         Returns {'risk': <low|moderate|high>, 'score': float}
#         """
#         text = (rag_answer or "") + "\n\n" + (retrieved_context or "")
#         text = text.strip() or "no content"
#
#         # Attempt ML model
#         if self.ml:
#             try:
#                 prob = float(self.ml.predict_proba([text])[0])  # 0..1
#                 # map prob to qualitative
#                 if prob >= 0.75:
#                     risk = "high"
#                 elif prob >= 0.4:
#                     risk = "moderate"
#                 else:
#                     risk = "low"
#                 return {"risk": risk, "score": prob, "method": "ml"}
#             except Exception as e:
#                 logger.warning(f"[PredictiveAgent] ML prediction failed: {e}")
#
#         # Fallback to LLM reasoning if available
#         if self.llm:
#             prompt = (
#                 "Given the following retrieved clinical context and model-generated answer, "
#                 "provide a short risk assessment (low/moderate/high) and a recommended plan. "
#                 "Return JSON with keys 'risk' and 'plan' and a numeric confidence 0..1.\n\n"
#                 f"Context:\n{retrieved_context}\n\nAnswer:\n{rag_answer}\n\nAssessment:"
#             )
#             resp = self.llm.llm.invoke([{"content": prompt}]) if hasattr(self.llm, "llm") else None
#             # The exact shape of response depends on ChatMistralAI wrapper. Use .content if present.
#             text_out = resp.content if resp is not None else ""
#             # Best-effort parsing: try to look for keywords. This is a fallback.
#             if "high" in text_out.lower():
#                 return {"risk": "high", "score": 0.8, "method": "llm"}
#             if "moderate" in text_out.lower():
#                 return {"risk": "moderate", "score": 0.5, "method": "llm"}
#             return {"risk": "low", "score": 0.2, "method": "llm"}
#
#         # final fallback: simple rules
#         if "blood sugar" in text.lower() or "hba1c" in text.lower() or "fasting" in text.lower():
#             return {"risk": "moderate", "score": 0.5, "method": "rule"}
#         return {"risk": "low", "score": 0.1, "method": "rule"}
#
#     def generate_plan(self, risk: str) -> Dict[str, Any]:
#         """Return a human-readable plan based on risk."""
#         if risk == "high":
#             plan = "Advise urgent clinical assessment and blood tests (FPG/A1c), consider referral."
#         elif risk == "moderate":
#             plan = "Recommend prompt GP visit for blood tests and lifestyle changes; follow-up in 4 weeks."
#         else:
#             plan = "Advise lifestyle advice; monitor symptoms and re-check if worse."
#         return {"plan": plan}
