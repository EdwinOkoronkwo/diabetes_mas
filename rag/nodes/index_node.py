import logging

from sentence_transformers import SentenceTransformer

from rag.indexer.ChromaIndexer import ChromaIndexer
from rag.state.RagAgentState import RagAgentState
from rag.utils.EmbeddingWrapper import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

def index_node(state: RagAgentState) -> RagAgentState:
    if not state.documents:
        logger.warning("[index_node] No documents found")
        return state

    # 1. Build embedding function
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_fn = SentenceTransformerEmbeddingFunction(embedding_model)

    # 2. Initialize indexer correctly
    indexer = ChromaIndexer(
        collection_name="medical_docs",
        embedding_fn=embedding_fn
    )

    # 3. Index documents
    for i, doc in enumerate(state.documents):
        indexer.add_document(str(i), doc.page_content)

    logger.info(f"[index_node] Indexed {len(state.documents)} documents")
    return state


# def index_node(state: RagAgentState) -> RagAgentState:
#     if not state.documents or not state.indexer:
#         logger.warning("[index_node] Documents or indexer not available")
#         return state
#
#     for i, doc in enumerate(state.documents):
#         state.indexer.add_document(str(i), doc.page_content)
#     logger.info(f"[index_node] Indexed {len(state.documents)} documents")
#     return state



# import logging
# from rag.indexer.ChromaIndexer import ChromaIndexer
# from rag.state.RagAgentState import RagAgentState
#
# logger = logging.getLogger(__name__)
#
# def index_node(state: RagAgentState) -> RagAgentState:
#     if not state.documents or not state.indexer:
#         logger.warning("[index_node] No documents or indexer available")
#         return state
#
#     for i, doc in enumerate(state.documents):
#         logger.info(f"[index_node] Adding doc {i}")
#         state.indexer.add_document(str(i), doc.page_content)
#     logger.info(f"[index_node] Indexed {len(state.documents)} documents")
#     return state


