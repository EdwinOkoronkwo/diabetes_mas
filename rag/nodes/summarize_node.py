import logging

from interfaces.rag.IGenerator import IGenerator
from rag.state.RagAgentState import RagAgentState


logger = logging.getLogger(__name__)

def summarize_node(state: RagAgentState) -> RagAgentState:
    generator: IGenerator = state.generator
    if not generator or not state.answer:
        logger.warning("[summarize_node] Generator or answer not available")
        state.summary = ""
        return state

    # Summarize using the generator interface
    state.summary = generator.summarize(state.answer)
    logger.info("[summarize_node] Summary generated")
    return state





# import logging
#
# from interfaces.rag.IGenerator import IGenerator
# from rag.state.RagAgentState import RagAgentState
#
#
# logger = logging.getLogger(__name__)
#
# def summarize_node(state: RagAgentState) -> RagAgentState:
#     generator: IGenerator = state.generator
#     answer: str = state.answer
#
#     if not generator or not answer:
#         logger.warning("[summarize_node] Generator or answer not available")
#         state.summary = ""
#         return state
#
#     # Use the generator interface to summarize
#     state.summary = generator.summarize(answer)
#     logger.info("[summarize_node] Summary generated")
#     return state


