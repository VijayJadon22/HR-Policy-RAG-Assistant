"""STEP 7: build the agent that ties the LLM and search tool together."""

from langchain.agents import create_agent
from hr_policy_rag_assistant.config import SYSTEM_PROMPT


def create_hr_agent(llm, tools):
    """Return a Langchain agent that can call our tools to answer the question."""
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
