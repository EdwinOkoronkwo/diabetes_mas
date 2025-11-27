from interfaces.rag.IGenerator import IGenerator
from rag.generator.MockLLMGenerator import MockLLMGenerator


import logging
import os

from rag.nodes.load_pdfs_node import load_pdfs_node
from rag.nodes.index_node import index_node
from rag.nodes.retrieval_node import retrieval_node
from rag.nodes.generate_node import generate_node
from rag.nodes.summarize_node import summarize_node
from rag.indexer.ChromaIndexer import ChromaIndexer
from rag.retriever.ChromaRetriever import ChromaRetriever
from rag.generator.LLMGenerator import LLMGenerator

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

from rag.state.RagAgentState import RagAgentState
from rag.utils.EmbeddingWrapper import SentenceTransformerEmbeddingFunction


def main():
    # Initialize components
    # RAG configuration
    # ---------------------------
    pdf_folder = "../data/01_raw/"
    query = "What is Meta GPT?"
    top_k = 5

    # ---------------------------
    # Load embedding model
    # ---------------------------
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding_fn = SentenceTransformerEmbeddingFunction(embed_model)
    retriever = ChromaRetriever("rag_docs", embedding_fn)
    generator: IGenerator = MockLLMGenerator()  # use interface type hint
    indexer = ChromaIndexer("rag_docs", embedding_fn)

    state = {
        "pdf_folder": "../data/01_raw/",
        "query": "What is Meta GPT?",
        "retriever": retriever,
        "generator": generator,
        "indexer": indexer,
        "top_k": 5
    }

    # Run RAG pipeline as usual
    state = load_pdfs_node(state)
    state = index_node(state)
    state = retrieval_node(state)
    state = generate_node(state)
    state = summarize_node(state)

    print("\n=== RAG ANSWER ===")
    print(state["answer"])
    print("\n=== SUMMARY ===")
    print(state["summary"])

# ---------------------------
# Run main
# ---------------------------
if __name__ == "__main__":
    main()