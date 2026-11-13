
from dataclasses import dataclass

from langchain_core.documents import Document

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
    """Load a document while preventing duplicate processing."""

    # Step 1: Generate stable document identity.
    document_id = compute_file_hash(file_path)

    # Step 2: Skip documents already registered.
    if registry.is_registered(document_id):
        return IngestionResult(
            document_id=document_id,
            documents=[],
            skipped=True,
        )

    # Step 3: Load the document.
    documents = load_document(file_path)

    # Step 4: Attach document identity to page-level metadata.
    for document in documents:
        document.metadata["document_id"] = document_id

    # Step 5: Register only after successful loading.
    registry.register(document_id)

    return IngestionResult(
        document_id=document_id,
        documents=documents,
        skipped=False,
    )
