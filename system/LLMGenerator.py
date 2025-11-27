import os
import logging
import json
from typing import List, Dict, Any, Type
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, create_model

# Assuming ILLMGenerator is the abstract interface
from interfaces.system.ILLMGenerator import ILLMGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMGenerator(ILLMGenerator):
    """
    Implements IGenerator using LangChain's ChatMistralAI, focused on
    RAG-style generation, summarization, and structured output.

    NOTE: Implements all methods required by ILLMGenerator.
    """

    def __init__(self, api_key: str = None, model_name: str = "mistral-small-latest"):

        # 1. API Key Loading Logic (using dotenv if available)
        if api_key is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                logger.warning("python-dotenv not installed. API key must be in environment.")

            # Check environment variables
            api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found. Please set it in your environment or provide it directly.")

        # 2. Model Initialization
        self.model_name = model_name
        logger.info(f"[LLMGenerator] Loading Mistral model: {model_name}")

        # Instantiate the LangChain Mistral client
        self.llm = ChatMistralAI(
            mistral_api_key=api_key,
            model=model_name,
            # Use a low temperature for predictable RAG and summarization
            temperature=0.1
        )
        logger.info("[LLMGenerator] Mistral model loaded successfully.")

    def generate(self, query: str, docs: List[str]) -> str:
        """
        Generates a RAG-style response by combining documents and a query.
        """
        context = "\n---\n".join(docs)

        prompt = f"""
You are a concise, helpful assistant. Use the provided context to answer the user's query. 
If the context does not contain the answer, state that you cannot answer based on the provided information.

CONTEXT:
---
{context}
---

QUERY:
{query}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def summarize(self, text: str) -> str:
        """
        Generates a summary of the provided text.
        """
        prompt = f"Provide a concise summary of the following text:\n\nTEXT:\n{text}"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def invoke_structured(self, messages: List[Dict[str, str]], response_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implements structured output generation using LangChain's Pydantic binding.

        Args:
            messages: List of message dictionaries (e.g., [{"role": "user", "content": "..."}])
            response_schema: A dictionary defining the desired JSON output schema.

        Returns:
            A dictionary matching the response_schema structure.
        """
        # 1. Dynamically create a Pydantic model from the response_schema dict
        # We need to map the dictionary structure to Pydantic fields.
        fields = {}
        for key, value in response_schema.items():
            # Simplistic mapping: assuming type is implied by placeholder or documentation
            # For real usage, response_schema should be a valid JSON schema or Pydantic definition.
            # Here we assume all keys map to simple string/float types based on context.
            if key == 'confidence':
                fields[key] = (float, ...)
            elif key == 'risk':
                fields[key] = (str, "low")
            else:
                fields[key] = (str, "")

        OutputModel: Type[BaseModel] = create_model('OutputModel', **fields)

        # 2. Bind the LLM to force structured JSON output
        structured_llm = self.llm.with_structured_output(OutputModel)

        # 3. Prepare LangChain-compatible messages
        lc_messages = [
            HumanMessage(content=msg['content']) if msg['role'] == 'user' else BaseMessage(**msg)
            for msg in messages
        ]

        # 4. Invoke the structured LLM
        response_model = structured_llm.invoke(lc_messages)

        # 5. Convert the Pydantic object back to a standard dictionary
        return response_model.model_dump()

    def invoke(self, query: str, docs: List[str] = None) -> str:
        """
        The primary invocation method, acting as an alias for generate.
        """
        if docs is None:
            docs = []

        return self.generate(query, docs)