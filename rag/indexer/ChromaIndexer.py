# rag/indexer/ChromaIndexer.py
import logging
from langchain_community.vectorstores import Chroma
from rag.db.chroma_client import get_chroma_db

logger = logging.getLogger(__name__)


class ChromaIndexer:
    def __init__(self, collection_name: str, embedding_fn, chroma_db=None):
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        self.db = chroma_db if chroma_db else get_chroma_db(collection_name, embedding_fn=embedding_fn)
        logger.info(f"[Indexer] Initialized for collection: {self.collection_name}")

    def add_document(self, doc_id: str, text: str):
        logger.info(f"[Indexer] Adding doc {doc_id}")
        self.db.add_texts([text], metadatas=[{"doc_id": doc_id}])

    def add_documents(self, docs: list[str]):
        for i, text in enumerate(docs):
            self.add_document(str(i), text)


# class ChromaIndexer:
#     def __init__(self, collection_name: str, embedding_fn):
#         self.collection_name = collection_name
#         self.db = get_chroma_db()
#         self.embedding_fn = embedding_fn
#         # self.db = Chroma(
#         #     collection_name=self.collection_name,
#         #     embedding_function=self.embedding_fn,
#         #     persist_directory=r"C:\CentennialCollege\COMP248_AI_Software_Design\Project\diabetes_mas\chroma_db"
#         # )
#         logger.info(f"[Indexer] ChromaIndexer initialized for collection: {self.collection_name}")
#
#     def add_document(self, doc_id: str, text: str):
#         logger.info(f"[Indexer] Adding doc {doc_id}")
#         self.db.add_texts([text], metadatas=[{"doc_id": doc_id}])
#
#     def add_documents(self, docs: list[str]):
#         for i, text in enumerate(docs):
#             self.add_document(str(i), text)


