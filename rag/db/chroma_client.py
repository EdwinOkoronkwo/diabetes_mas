# rag/db/chroma_client.py
import logging
from chromadb import Client
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

from rag.utils.EmbeddingWrapper import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

def get_chroma_db(collection_name: str = "diabetes_docs", persist_dir: str = None, embedding_fn=None):
    if embedding_fn is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_fn = SentenceTransformerEmbeddingFunction(model)
    if persist_dir is None:
        persist_dir = r"C:\CentennialCollege\COMP248_AI_Software_Design\Project\diabetes_mas\chroma_db"

    logger.info(f"[ChromaDB] Initializing collection '{collection_name}' at '{persist_dir}'")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_dir
    )
