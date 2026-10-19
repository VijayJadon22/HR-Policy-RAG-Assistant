"""All the setting for the app lives here"""

import os
from dotenv import load_dotenv

load_dotenv()

# ENV variables
groq_key = os.getenv("GROQ_API_KEY")
jina_key = os.getenv("JINA_API_KEY")

# Define Path / Vector Store
DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")

VECTOR_STORE_PATH = os.path.join("data", "faiss_index")

# LLM and Embedding Model
LLM_MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"

# Chunk size and overlap config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval results
TOP_K_RESULTS = 3

# Sytem Prompt
SYSTEM_PROMPT = (
    "You are a friendly HR assistant,"
    "Always use the search_hr_policy tool to look up any hr related policy facts"
    "before answering, if the answer is not in search_results, say You Don't Know instead of guessing"
)

def check_api_keys()->None:
    if not groq_key:
        raise ValueError("Missing GROQ API key")
    if not jina_key:
        raise ValueError("Missing JINA API key")
