# diabetes_mas/agents/RetrievalAgent.py (Refined)

# agents/RetrievalAgent.py

from typing import Dict, Any
from rag.state.RagAgentState import RagAgentState

from rag.state.RagAgentState import RagAgentState
from rag.retriever.ChromaRetriever import ChromaRetriever
from rag.pipeline.run_rag_pipeline import run_rag_pipeline

from rag.state.RagAgentState import RagAgentState
from rag.retriever.ChromaRetriever import ChromaRetriever
from rag.indexer.ChromaIndexer import ChromaIndexer
from rag.generator.LLMGenerator import LLMGenerator
from rag.generator.MockLLMGenerator import MockLLMGenerator
from rag.pipeline.run_rag_pipeline import run_rag_pipeline
from sentence_transformers import SentenceTransformer
import logging
from rag.utils.EmbeddingWrapper import SentenceTransformerEmbeddingFunction

from rag.chroma_db import get_chroma_db
logger = logging.getLogger(__name__)

class RetrievalAgent:
    def __init__(self, retriever=None, indexer=None, generator=None, pipeline_callable=None, use_mock_llm=False):
        # Shared embedding function
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_fn = SentenceTransformerEmbeddingFunction(model)

        # Shared Chroma DB instance
        self.chroma_db = None  # Will be initialized by first component if not provided

        # Indexer
        if indexer is None:
            self.indexer = ChromaIndexer(
                collection_name="diabetes_docs_v2",
                embedding_fn=embedding_fn,
                chroma_db=self.chroma_db
            )
        else:
            self.indexer = indexer

        # Retriever
        if retriever is None:
            self.retriever = ChromaRetriever(
                collection_name="diabetes_docs_v2",
                embedding_fn=embedding_fn,
                chroma_db=self.indexer.db  # ensure same DB
            )
        else:
            self.retriever = retriever

        # Generator
        if generator is None:
            self.generator = MockLLMGenerator() if use_mock_llm else LLMGenerator()
        else:
            self.generator = generator

        # Pipeline callable
        self.pipeline = pipeline_callable if pipeline_callable else run_rag_pipeline

    def run(self, state):
        """
        Runs a fresh RAG pipeline for each query.
        Ensures no stale state or previous results are carried over.
        """


        query = state["current_input"]

        # 1️⃣ Create a fresh RAG state
        rag_state = RagAgentState(
            pdf_folder="rag/data/",
            query=query,
        )

        # Reset fields explicitly (defensive)
        rag_state.retrieved_docs.clear()
        rag_state.answer = ""
        rag_state.summary = ""
        rag_state.embedded_query = None

        # 2️⃣ Assign components
        rag_state.retriever = self.retriever
        rag_state.generator = self.generator
        rag_state.indexer = self.indexer

        # 3️⃣ Compute query embedding and store it
        if self.retriever and hasattr(self.retriever, "embedding_fn"):
            rag_state.embedded_query = self.retriever.embedding_fn.embed_query(query)

        # 4️⃣ Run the RAG pipeline
        final = self.pipeline(rag_state)

        # 5️⃣ Update MAS AgentState with results from this run
        state["retrieved_context"] = "\n\n".join(
            doc.page_content for doc in final.retrieved_docs
        )
        state["rag_answer"] = final.answer
        state["rag_summary"] = final.summary

        return state


# diabetes_mas/agents/RetrievalAgent.py

# from interfaces.agents.IRetrievalAgent import IRetrievalAgent
# from interfaces.rag.IRetriever import IRetriever
# from interfaces.rag.IGenerator import IGenerator
# from rag.retriever.ChromaRetriever import ChromaRetriever
# from rag.state.RagAgentState import RagAgentState
# from rag.pipeline.run_rag_pipeline import run_rag_pipeline  # Your unified RAG pipeline
#
# class RetrievalAgent(IRetrievalAgent):
#     def __init__(
#         self,
#         retriever: IRetriever = None,
#         generator: IGenerator = None,
#         pipeline_callable=run_rag_pipeline
#     ):
#         self.retriever = retriever if retriever else ChromaRetriever()
#         self.generator = generator
#         self.pipeline = pipeline_callable  # Accept any callable that takes RagAgentState
#
#     def query(self, query: str, pdf_folder: str = "../data/01_raw/") -> dict:
#         """
#         Run the full RAG pipeline internally and return structured results.
#         """
#         # 1. Initialize internal RAG state
#         state = RagAgentState(
#             pdf_folder=pdf_folder,
#             query=query,
#         )
#         # Assign components
#         state.retriever = self.retriever
#         state.generator = self.generator
#
#         # 2. Run the RAG pipeline
#         state = self.pipeline(state)
#
#         # 3. Return structured results
#         return {
#             "answer": state.answer,
#             "summary": state.summary,
#             "retrieved_docs": [doc.page_content for doc in state.retrieved_docs]
#         }
#
#     def retrieve_context(self, query: str) -> str:
#         """
#         Legacy method: retrieves only context string.
#         """
#         if not self.retriever:
#             return ""
#         embedded_query = self.retriever.embed_query(query)
#         return self.retriever.similarity_search(embedded_query)
#
#     def summarize_context(self, context: str) -> str:
#         """
#         Legacy method: summarizes context string.
#         """
#         if self.generator is None:
#             return f"Summary (Mocked): {context[:80]}..."
#         return self.generator.summarize(context)
