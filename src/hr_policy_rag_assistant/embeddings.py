"""STEP:3 Turn text into Numbers(vectors) using Jina"""

from langchain_community.embeddings import JinaEmbeddings
from hr_policy_rag_assistant.config import EMBEDDING_MODEL_NAME
from typing import List


def get_embeddings_model():
    """Return a JINA embedding model, Reads the Jina api key from the environment."""

    return JinaEmbeddings(model_name=EMBEDDING_MODEL_NAME)
