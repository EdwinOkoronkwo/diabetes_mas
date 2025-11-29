import logging
from typing import TypedDict

from sentence_transformers import SentenceTransformer

from rag.retriever.ChromaRetriever import ChromaRetriever, logger
from rag.state.RagAgentState import RagAgentState
from rag.utils.EmbeddingWrapper import SentenceTransformerEmbeddingFunction
from system.AgentState import AgentState
# Import necessary dependencies for creating the embedding function
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)



def retrieval_node(state: RagAgentState) -> RagAgentState:
    query = getattr(state, "query", "")
    
    # Use existing retriever from state if available
    if hasattr(state, "retriever") and state.retriever:
        retriever = state.retriever
    else:
        # Fallback
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_fn = SentenceTransformerEmbeddingFunction(embedding_model)
        retriever = ChromaRetriever(
            collection_name="diabetes_docs_v2",
            embedding_fn=embedding_fn
        )

    # Search by query text only
    logger.info("[retrieval_node] Searching by raw query text")
    docs = retriever.similarity_search(query)

    # Store results in state
    state.retrieved_docs = docs
    state.retrieved_context = "\n\n".join(doc.page_content for doc in docs)

    logger.info(f"[retrieval_node] Retrieved {len(docs)} docs")

    return state




