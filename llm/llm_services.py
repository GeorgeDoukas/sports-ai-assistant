import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# --- Centralized Configuration Loading ---
load_dotenv()

# LLM Provider Config
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b-it-qat")
LANGUAGE = os.getenv("LANGUAGE", "English")


# --- Centralized Pydantic Models ---
class ArticleSummary(BaseModel):
    """Data model for a final, verified summary created in a single pass."""

    summary: str = Field(description="The final, objective summary of 3-6 sentences.")
    highlights: list[str] = Field(
        description="The final, complete list of 4-10 key highlights."
    )


# --- Centralized, Flexible LLM Loader ---
def get_llm():
    """
    Initializes and returns the correct LLM provider based on the .env file.

    Args:
        model_type (str): "main" for the primary model, "fact_checker" for a more powerful one.
    """
    provider = LLM_PROVIDER.lower()

    # Announce which model is being initialized
    print(f"ℹ️  Initializing LLM ({LLM_MODEL}) via provider: {provider}")

    if provider == "openai_compatible":
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=f"{os.getenv("OPENAI_API_BASE").rstrip("/")}/v1",
        )
    elif provider == "ollama":
        return ChatOllama(model=LLM_MODEL)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}.")
