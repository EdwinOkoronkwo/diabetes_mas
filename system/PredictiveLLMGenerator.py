import aiohttp

from interfaces.system.AbstractLLMInterpreter import AbstractLLMInterpreter
import logging
import os
from typing import Dict, Any, Optional, List, Type
from abc import ABC, abstractmethod
import logging
import os
import time
import json
import asyncio
from pprint import pprint
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


# --- Configuration and Setup ---

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ["MISTRAL_API_KEY"] = "ECnNejAzltaN1KG0zWdpNtSU9y7Wsezv"
os.environ["MISTRAL_MODEL"] = "mistral-small-latest"

# --- API Constants ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


# -------------------------------------------------------------------------
# Predictive LLM Generator (NO LANGCHAIN)
# -------------------------------------------------------------------------

class PredictiveLLMGenerator(AbstractLLMInterpreter):
    """
    Direct Mistral API client using aiohttp.
    Handles retries, JSON response parsing, and fallback logic.
    """

    def __init__(self):
        self.max_retries = 3
        logger.info("[PredictiveLLMGenerator] Mistral API client initialized.")

        if not MISTRAL_API_KEY:
            logger.error("MISTRAL_API_KEY missing! LLM calls cannot succeed.")

    async def interpret(self, context: str, ml_score: float) -> Dict[str, Any]:

        # Prompt refinement: Emphasize the importance of the ML score in the final risk determination
        system_prompt = (
            "You are a Senior Clinical Systems Analyst combining RAG context "
            "with an ML risk score. If the ML score is >= 0.7 and clinical context "
            "is alarming, the riskLevel must be 'high'. Return ONLY a valid JSON object."
        )

        user_query = f"""
        Interpret the following data:
        - ML Risk Score: {ml_score:.3f}
        - Clinical context: {context}

        Output JSON with:
        - riskLevel: high | moderate | low
        - plan: detailed clinical action plan (as a dictionary with keys: immediateActions, medicationManagement, lifestyleInterventions, followUp)
        """

        payload = {
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        # --- Retry Loop with aiohttp ---
        for attempt in range(self.max_retries):
            try:
                # Use aiohttp.ClientSession for standard async Python requests
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        MISTRAL_API_URL,
                        headers=headers,
                        data=json.dumps(payload)
                    ) as response:

                        if response.status == 200:
                            result = await response.json()
                            json_str = result["choices"][0]["message"]["content"]
                            return json.loads(json_str)

                        logger.warning(
                            f"Mistral error {response.status}. "
                            f"Retrying in {2**attempt}s..."
                        )
                        await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Mistral call failed attempt {attempt+1}: {e}")
                await asyncio.sleep(2 ** attempt)

        # ---------------------------------------------------------------------
        # Fallback Logic
        # ---------------------------------------------------------------------
        logger.error("Mistral API failed after retries. Returning fallback.")

        if ml_score >= 0.7:
            risk = "high"
        elif ml_score >= 0.4:
            risk = "moderate"
        else:
            risk = "low"

        return {
            "riskLevel": risk,
            "plan": "LLM unavailable. Risk based only on ML model thresholds."
        }





# class PredictiveLLMGenerator(AbstractLLMInterpreter):
#     """
#     Direct Mistral API client.
#     Handles retries, JSON response parsing, and fallback logic.
#     """
#
#     def __init__(self):
#         self.max_retries = 3
#         logger.info("[PredictiveLLMGenerator] Mistral API client initialized.")
#
#         if not MISTRAL_API_KEY:
#             logger.error("MISTRAL_API_KEY missing! LLM calls cannot succeed.")
#
#     async def interpret(self, context: str, ml_score: float) -> Dict[str, Any]:
#
#         system_prompt = (
#             "You are a Senior Clinical Systems Analyst combining RAG context "
#             "with an ML risk score. Return ONLY a valid JSON object."
#         )
#
#         user_query = f"""
#         Interpret the following:
#         - ML Risk Score: {ml_score:.3f}
#         - Clinical context: {context}
#
#         Output JSON with:
#         - riskLevel: high | moderate | low
#         - plan: detailed clinical action plan
#         """
#
#         payload = {
#             "model": MISTRAL_MODEL,
#             "messages": [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_query}
#             ],
#             "response_format": {"type": "json_object"}
#         }
#
#         headers = {
#             "Authorization": f"Bearer {MISTRAL_API_KEY}",
#             "Content-Type": "application/json"
#         }
#
#         # --- Retry Loop ---
#         for attempt in range(self.max_retries):
#             try:
#                 async with aiohttp.ClientSession() as session:
#                     async with session.post(
#                         MISTRAL_API_URL,
#                         headers=headers,
#                         data=json.dumps(payload)
#                     ) as response:
#
#                         if response.status == 200:
#                             result = await response.json()
#                             json_str = result["choices"][0]["message"]["content"]
#                             return json.loads(json_str)
#
#                         logger.warning(
#                             f"Mistral error {response.status}. "
#                             f"Retrying in {2**attempt}s..."
#                         )
#                         await asyncio.sleep(2 ** attempt)
#
#             except Exception as e:
#                 logger.error(f"Mistral call failed attempt {attempt+1}: {e}")
#                 await asyncio.sleep(2 ** attempt)
#
#         # ---------------------------------------------------------------------
#         # Fallback Logic
#         # ---------------------------------------------------------------------
#         logger.error("Mistral API failed after retries. Returning fallback.")
#
#         if ml_score >= 0.7:
#             risk = "high"
#         elif ml_score >= 0.4:
#             risk = "moderate"
#         else:
#             risk = "low"
#
#         return {
#             "riskLevel": risk,
#             "plan": "LLM unavailable. Risk based only on ML model thresholds."
#         }