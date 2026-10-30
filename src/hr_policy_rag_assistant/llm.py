"""Step 6: Connect to the LLM (the "brain" of the assistant)"""

from langchain_groq import ChatGroq
from hr_policy_rag_assistant.config import LLM_MODEL_NAME


def get_llm():
    """Return a Groq chat model. Reads GROQ API KEY from the environment."""
    return ChatGroq(model=LLM_MODEL_NAME, temperature=0)
