import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


def _generate_chunk_id(
    document_id: str,
    segment_identifier: str,
    chunk_index: int,
    content: str,
) -> str:
    """Generate a deterministic SHA-256 ID for a document chunk."""

    identity = (
        f"{document_id}:"
        f"{segment_identifier}:"
        f"{chunk_index}:"
        f"{content}"
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split LangChain Documents into smaller retrieval-ready chunks.

    Each input Document is split independently to preserve its
    original source metadata and page-level provenance.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunked_documents = []

    for document in documents:

        # Ignore empty and whitespace-only documents.
        if not document.page_content.strip():
            continue

        # Split one source Document at a time.
        chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(chunks):

            # Preserve all original metadata.
            metadata = dict(document.metadata)

            document_id = metadata.get("document_id", "")
            page_number = metadata.get("page_number")

            # PDF pages and TXT documents need distinct segment identities.
            if page_number is not None:
                segment_identifier = f"page:{page_number}"
            else:
                segment_identifier = "document"

            chunk_id = _generate_chunk_id(
                document_id=document_id,
                segment_identifier=segment_identifier,
                chunk_index=chunk_index,
                content=chunk.page_content,
            )

            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                }
            )

            chunked_documents.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )

    return chunked_documents
