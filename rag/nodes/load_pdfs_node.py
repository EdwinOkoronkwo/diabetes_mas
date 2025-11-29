import os
import logging
from pypdf import PdfReader
from langchain_core.documents import Document
from rag.state.RagAgentState import RagAgentState

logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs_node(state: RagAgentState) -> RagAgentState:
    if not state.pdf_folder or not os.path.exists(state.pdf_folder):
        logger.warning(f"[load_pdfs_node] PDF folder not found: {state.pdf_folder}")
        state.documents = []
        return state

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Characters per chunk
        chunk_overlap=200,  # Overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []
    for file_name in os.listdir(state.pdf_folder):
        if not file_name.lower().endswith(".pdf"):
            continue
        path = os.path.join(state.pdf_folder, file_name)
        reader = PdfReader(path)

        # Extract all text first
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()

        # Split into chunks
        chunks = text_splitter.create_documents(
            [full_text],
            metadatas=[{"source": file_name}]
        )
        documents.extend(chunks)

    state.documents = documents
    logger.info(f"[load_pdfs_node] Loaded {len(documents)} document chunks from PDFs")
    return state



# import os
# import logging
# from pypdf import PdfReader
# from langchain_core.documents import Document
#
# from rag.state.RagAgentState import RagAgentState
#
# logger = logging.getLogger(__name__)
#
# def load_pdfs_node(state: RagAgentState) -> RagAgentState:
#     pdf_folder = state.pdf_folder
#     if not pdf_folder or not os.path.exists(pdf_folder):
#         logger.warning(f"PDF folder not found: {pdf_folder}")
#         state.documents = []
#         return state
#
#     documents = []
#     for file_name in os.listdir(pdf_folder):
#         if not file_name.lower().endswith(".pdf"):
#             continue
#         path = os.path.join(pdf_folder, file_name)
#         reader = PdfReader(path)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() or ""
#         documents.append(Document(page_content=text))
#     logger.info(f"[load_pdfs_node] Loaded {len(documents)} PDFs")
#     state.documents = documents
#     return state



