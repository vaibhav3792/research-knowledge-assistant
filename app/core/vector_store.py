from functools import lru_cache

from langchain_chroma import Chroma

from app.core.config import settings
from app.core.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Create and cache the persistent Chroma vector store.

    Uses the configured embedding model, persistence directory,
    and collection name.
    """

    # Ensure the persistence directory exists.
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )
