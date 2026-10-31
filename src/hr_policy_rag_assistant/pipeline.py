""" "Wires all the component together into one ready to use agent
This is the one file that main.py (CLI) and app.py (streamlit) both call.
Each step is handled by its own small module
"""

from hr_policy_rag_assistant.config import DATA_DIR
from hr_policy_rag_assistant.config import LLM_MODEL_NAME
from hr_policy_rag_assistant.agent import create_hr_agent
from hr_policy_rag_assistant.document_loader import load_all_documents
from hr_policy_rag_assistant.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriever,
)


def build_vector_store_for_documents(file_path: str = DATA_DIR):
    """Load + Split + embed the documents, reusing a saved index if we have one."""
    if vector_store_exists():
        print("Found a saved vector store on disk, loading it(fast, no re-embedding).")
        return load_vector_store()
    print("No vector store found, building one from scratch.")
    documents = load_all_documents()
    
