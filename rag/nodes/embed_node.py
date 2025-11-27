import logging
from rag.state.RagAgentState import RagAgentState

logger = logging.getLogger(__name__)

def embed_node(state: RagAgentState) -> RagAgentState:
    if not state.documents or not state.indexer:
        logger.warning("[embed_node] Documents or indexer not available")
        return state

    for doc in state.documents:
        state.indexer.embed_document(doc.page_content)
    logger.info("[embed_node] Embedded documents")
    return state



# import logging
#
# from rag.state.RagAgentState import RagAgentState
#
# logger = logging.getLogger(__name__)
#
# def embed_node(state: RagAgentState) -> RagAgentState:
#     if not state.documents or not state.indexer:
#         logger.warning("[embed_node] Documents or indexer not available")
#         return state
#
#     # Compute embeddings for all docs (assumes indexer has embed_document method)
#     for doc in state.documents:
#         state.indexer.embed_document(doc.page_content)
#     logger.info("[embed_node] Embedded documents")
#     return state


