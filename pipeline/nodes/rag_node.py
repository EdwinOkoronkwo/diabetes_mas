import logging
from rag.retriever.ChromaRetriever import ChromaRetriever, logger
from rag.utils.EmbeddingWrapper import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from typing import List

COLLECTION_NAME = "diabetes_docs_v2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize embedding function once
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_embedding_fn = SentenceTransformerEmbeddingFunction(_model)

def rag_node(state: dict) -> dict:
    """
    Full RAG node: retrieves documents and synthesizes answer/summary
    """
    query = state.get("current_input", "")

    # Initialize retriever
    retriever = ChromaRetriever(collection_name=COLLECTION_NAME, embedding_fn=_embedding_fn)

    # Retrieve documents (no need to store embedding for now)
    docs = retriever.similarity_search(query)
    state["retrieved_docs"] = docs
    state["retrieved_context"] = "\n\n".join(doc.page_content for doc in docs)

    # --- Placeholder for answer generation ---
    # You can plug in your LLM generator here (Mistral or mock)
    llm_answer = f"Synthesized answer for: {query}"
    llm_summary = f"Summary of retrieved context for: {query}"

    state["rag_answer"] = llm_answer
    state["rag_summary"] = llm_summary

    logger.info(f"[rag_node] Retrieved {len(docs)} docs and generated answer/summary")
    return state
