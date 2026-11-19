
from langchain_core.documents import Document


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into readable, source-aware context."""

    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    if not documents:
        return ""

    formatted_chunks = []

    for index, document in enumerate(documents, start=1):
        if not isinstance(document, Document):
            raise TypeError(
                f"Item at position {index} must be a LangChain Document"
            )

        if not document.page_content.strip():
            continue

        metadata = document.metadata

        chunk_id = metadata.get("chunk_id", "Unknown")
        document_id = metadata.get("document_id", "Unknown")
        filename = metadata.get("filename", metadata.get("source", "Unknown"))
        page_number = metadata.get("page_number", "Unknown")

        chunk = (
            f"[SOURCE: {chunk_id}]\n"
            f"Document ID: {document_id}\n"
            f"Filename: {filename}\n"
            f"Page: {page_number}\n\n"
            f"{document.page_content.strip()}"
        )

        formatted_chunks.append(chunk)

    return "\n\n---\n\n".join(formatted_chunks)
