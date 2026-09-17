import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


load_dotenv()


def get_llm():
    """Create and return the OpenRouter LLM."""

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set in the .env file."
        )

    llm = ChatOpenRouter(
        model="openai/gpt-5-mini",
        temperature=0,
        max_tokens=1000,
        api_key=api_key,
    )

    return llm