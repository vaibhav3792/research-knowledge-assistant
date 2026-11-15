from langchain_core.documents import Document

from app.core.vector_store import get_vector_store


def index_chunks(
    chunks: list[Document],
) -> list[str]:
    """
    Store document chunks in the persistent vector database.

    Each chunk's deterministic chunk_id is used as its Chroma record ID.
    Re-indexing the same chunk therefore updates its existing record.
    """

    # Nothing to index.
    if not chunks:
        return []

    # Extract and validate deterministic chunk IDs.
    chunk_ids = []

    for index, chunk in enumerate(chunks):
        chunk_id = chunk.metadata.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Chunk at position {index} has no valid chunk_id."
            )

        chunk_ids.append(chunk_id)

    # Prevent duplicate IDs within one indexing batch.
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs found in the indexing batch."
        )

    # Get the persistent vector store.
    vector_store = get_vector_store()

    # Add or update documents using their deterministic IDs.
    return vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )
