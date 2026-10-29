from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from hr_policy_rag_assistant.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: List[Document]) -> List[Document]:
    """Split loaded documents into smaller chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    return chunks
