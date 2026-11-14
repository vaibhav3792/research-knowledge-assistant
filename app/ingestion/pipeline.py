
from dataclasses import dataclass

from langchain_core.documents import Document

from app.ingestion.chunker import chunk_documents
from app.ingestion.document_identity import compute_file_hash
from app.ingestion.document_registry import DocumentRegistry
from app.ingestion.loader import load_document


@dataclass
class IngestionResult:
    document_id: str
    documents: list[Document]
    skipped: bool


def ingest_document(
    file_path: str,
    registry: DocumentRegistry,
) -> IngestionResult:
    """Load, chunk, and register a document while preventing duplicates."""

    # Step 1: Generate stable document identity from the original file.
    document_id = compute_file_hash(file_path)

    # Step 2: Skip documents already registered.
    if registry.is_registered(document_id):
        return IngestionResult(
            document_id=document_id,
            documents=[],
            skipped=True,
        )

    # Step 3: Load the original document.
    page_documents = load_document(file_path)

    # Step 4: Attach document identity to page-level metadata.
    for document in page_documents:
        document.metadata["document_id"] = document_id

    # Step 5: Split page-level documents into retrieval-ready chunks.
    chunks = chunk_documents(page_documents)

    # Step 6: Register only after loading and chunking succeed.
    registry.register(document_id)

    # Step 7: Return the chunk-level documents.
    return IngestionResult(
        document_id=document_id,
        documents=chunks,
        skipped=False,
    )
