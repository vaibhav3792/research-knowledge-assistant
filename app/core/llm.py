from functools import lru_cache

from langchain_groq import ChatGroq

from app.core.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Create and cache the configured Groq chat model."""

    if settings.groq_api_key is None:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Please set it in your .env file."
        )

    return ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.groq_api_key.get_secret_value(),
    )
