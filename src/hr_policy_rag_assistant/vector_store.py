"""Store chunk embeddings in FAISS so we can search them later."""

from langchain_community.vectorstores import FAISS
from hr_policy_rag_assistant.config import VECTOR_STORE_PATH
from hr_policy_rag_assistant.embeddings import get_embeddings_model


def build_vector_store(chunks):
    """Embed every chunk and build a searchable FAISS index memory."""
    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks, embeddings_model)


def save_vector_store(vector_store, path: str = VECTOR_STORE_PATH) -> None:
    """Save the FAISS indexto index so that we dont have to load it every time"""
    vector_store.save_local(path)


def load_vector_store(path: str = VECTOR_STORE_PATH):
    """Load a previously stored FAISS index from disk."""
    embeddings_model = get_embeddings_model()
    return FAISS.load_local(
        path, embeddings_model, allow_dangerous_deserialization=True
    )
