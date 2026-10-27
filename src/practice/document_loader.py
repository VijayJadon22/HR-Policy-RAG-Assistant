from langchain_community.document_loaders import TextLoader, PyPDFLoader
from config import DATA_DIR
from typing import List
from langchain_core.documents import Document


def load_all_documents() -> List[Document]:
    documents: List[Document] = []

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Load all PDF files
    pdf_file_paths = list(DATA_DIR.glob("**/*.pdf"))

    for file_path in pdf_file_paths:
        loader = PyPDFLoader(str(file_path))
        loaded_docs = loader.load()
        documents.extend(loaded_docs)

    # Load all TXT files
    text_file_paths = list(DATA_DIR.glob("**/*.txt"))

    for file_path in text_file_paths:
        loader = TextLoader(str(file_path), encoding="utf-8")
        loaded_docs = loader.load()
        documents.extend(loaded_docs)

    return documents
