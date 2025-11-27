import logging
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

from interfaces.rag.IGenerator import IGenerator

logger = logging.getLogger(__name__)

# rag/generator/LLMGenerator.py
import os
import logging
from typing import List
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from langchain_mistralai import ChatMistralAI
except Exception:
    # If the langchain_mistralai package is unavailable, raise a clear error
    raise



#
class LLMGenerator(IGenerator):
    def __init__(self, api_key: str = None, model_name: str = "mistral-small-latest"):
        import os
        from langchain_mistralai import ChatMistralAI
        import logging

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Use provided key or .env / environment variable
        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set in environment or .env file")

        logger.info(f"[LLM] Loading Mistral model: {model_name}")
        self.llm = ChatMistralAI(api_key=api_key, model=model_name)
        logger.info("[LLM] Mistral model loaded")

    def generate(self, query: str, docs: list[str]) -> str:
        context = "\n\n".join(docs)
        prompt = context
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def summarize(self, text: str) -> str:
        prompt = f"Summarize this text:\n{text}"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def invoke(self, query: str, docs: list[str]) -> str:
        return self.generate(query, docs)



# class LLMGenerator:
#     """
#     Thin wrapper around Mistral via langchain_mistralai.ChatMistralAI.
#     Provides `generate(query, docs)` which returns a string answer.
#     """
#
#     def __init__(self, api_key: str = None, model_name: str = None):
#         # Load from env if not provided
#         if api_key is None:
#             from dotenv import load_dotenv
#             load_dotenv()
#             api_key = os.getenv("MISTRAL_API_KEY")
#
#         if model_name is None:
#             from dotenv import load_dotenv
#             load_dotenv()
#             model_name = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
#
#         if not api_key:
#             raise ValueError("MISTRAL_API_KEY not set in environment or .env file")
#
#         logger.info(f"[LLM] Loading Mistral model: {model_name}")
#         self.llm = ChatMistralAI(api_key=api_key, model=model_name)
#         logger.info("[LLM] Mistral model loaded")
#
#     def generate(self, query: str, docs: List[str]) -> str:
#         """
#         docs: list of retrieved text passages (strings)
#         query: original user query
#         """
#         # Build a prompt that instructs the model to rely only on context (useful for RAG)
#         context = "\n\n".join(docs) if docs else ""
#         prompt = (
#             "You are an assistant that MUST answer using only the context below.\n"
#             + "If the context does not have the information, respond: "
#             + "\"The provided documents do not contain information about this topic.\"\n\n"
#             + f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
#         )
#
#         response = self.llm.invoke([HumanMessage(content=prompt)])
#         # ChatMistralAI returns an object with .content
#         return response.content
#
#     # Optional: summarization helper
#     def summarize(self, text: str) -> str:
#         prompt = f"Summarize the following text in plain language:\n\n{text}"
#         response = self.llm.invoke([HumanMessage(content=prompt)])
#         return response.content