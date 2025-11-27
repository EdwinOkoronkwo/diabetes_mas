from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
from rag.nodes.index_node import SentenceTransformerEmbeddingFunction


def get_chroma_db():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_fn = SentenceTransformerEmbeddingFunction(model)

    persist_dir = r"C:\CentennialCollege\COMP248_AI_Software_Design\Project\diabetes_mas\chroma_db"

    return Chroma(
        collection_name="diabetes_docs",
        embedding_function=embedding_fn,
        persist_directory=persist_dir
    )
