


# rag/pipeline/run_rag_pipeline.py

from rag.nodes.load_pdfs_node import load_pdfs_node
from rag.nodes.index_node import index_node
from rag.nodes.retrieval_node import retrieval_node
from rag.nodes.generate_node import generate_node
from rag.nodes.summarize_node import summarize_node


def run_rag_pipeline(state):
    """Pure functional RAG pipeline."""
    state = load_pdfs_node(state)
    state = index_node(state)
    state = retrieval_node(state)
    state = generate_node(state)
    state = summarize_node(state)
    return state


# import logging
# import os
#
# from rag.nodes.load_pdfs_node import load_pdfs_node
# from rag.nodes.index_node import index_node
# from rag.nodes.retrieval_node import retrieval_node
# from rag.nodes.generate_node import generate_node
# from rag.nodes.summarize_node import summarize_node
# from rag.indexer.ChromaIndexer import ChromaIndexer
# from rag.retriever.ChromaRetriever import ChromaRetriever
# from rag.generator.LLMGenerator import LLMGenerator
#
# from sentence_transformers import SentenceTransformer
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_mistralai import ChatMistralAI
# from langchain_core.messages import HumanMessage
#
# from rag.state.RagAgentState import RagAgentState
#
# import os
# from dotenv import load_dotenv
#
# from rag.utils.EmbeddingWrapper import SentenceTransformerEmbeddingFunction
#
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
#
#
#
# # ---------------------------
# # Choose generator here
# # ---------------------------
# USE_MOCK = True  # Set False to use real Mistral LLM
#
# if USE_MOCK:
#     from rag.generator.MockLLMGenerator import MockLLMGenerator
#     generator = MockLLMGenerator()
# else:
#     from rag.generator.LLMGenerator import LLMGenerator
#     generator = LLMGenerator(model_name="mistral-small-latest")
#
#
#
# # ---------------------------
# # Main function
# # ---------------------------
# def main():
#     pdf_folder = "../data/01_raw/"
#     query = "What is Meta GPT?"
#     top_k = 5
#
#     embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     embedding_fn = SentenceTransformerEmbeddingFunction(embed_model)
#
#     indexer = ChromaIndexer("rag_docs", embedding_fn)
#     retriever = ChromaRetriever("rag_docs", embedding_fn)
#
#     # LLMGenerator handles API key internally
#     llm = LLMGenerator(model_name="mistral-small-latest")
#
#     # ---------------------------
#     # Create RAG state
#     # ---------------------------
#     state = RagAgentState(
#         pdf_folder=pdf_folder,
#         query=query,
#         indexer=indexer,
#         retriever=retriever,
#         generator=generator,  # can be mock or real
#         top_k=top_k
#     )
#
#     # ---------------------------
#     # Run pipeline nodes
#     # ---------------------------
#     state = load_pdfs_node(state)
#     state = index_node(state)
#     state = retrieval_node(state)
#     state = generate_node(state)
#     state = summarize_node(state)
#
#     # ---------------------------
#     # Output results
#     # ---------------------------
#     print("\n=== RAG ANSWER ===\n")
#     print(state.answer)
#     print("\n=== SUMMARY ===\n")
#     print(state.summary)
#
#
# # ---------------------------
# # Run main
# # ---------------------------
# if __name__ == "__main__":
#     main()


