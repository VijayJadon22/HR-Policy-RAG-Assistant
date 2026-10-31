""" "Wires all the component together into one ready to use agent
This is the one file that main.py (CLI) and app.py (streamlit) both call.
Each step is handled by its own small module
"""

from hr_policy_rag_assistant.config import DATA_DIR, LLM_MODEL_NAME, check_api_keys
from hr_policy_rag_assistant.agent import create_hr_agent
from hr_policy_rag_assistant.document_loader import load_all_documents
from hr_policy_rag_assistant.splitter import split_documents
from hr_policy_rag_assistant.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriever,
)
from hr_policy_rag_assistant.tools import create_search_tool
from hr_policy_rag_assistant.llm import get_llm


def build_vector_store_for_documents(file_path: str = DATA_DIR):
    """Load + Split + embed the documents, reusing a saved index if we have one."""
    if vector_store_exists():
        print("Found a saved vector store on disk, loading it(fast, no re-embedding).")
        return load_vector_store()
    print("No vector store found, building one from scratch.")
    documents = load_all_documents()
    chunks = split_documents(documents)
    print(f"Loaded '{file_path}' and split into {len(chunks)} chunks.")

    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    print("Vector store built and saved to disk for next time.")
    return vector_store


def build_hr_assistant(file_path: str = DATA_DIR):
    """Build the full RAG agent, ready to answer the question."""
    check_api_keys()

    vector_store = build_vector_store_for_documents()
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)

    llm = get_llm()
    agent = create_hr_agent(llm, [search_tool])

    return agent


def ask_agent(agent, question: str) -> str:
    """Ask the agent a question and return its final answer as plain text."""
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content
