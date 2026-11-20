
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api import main
from app.schemas.answer import AnswerResponse, Source

from pathlib import Path
import hashlib

from app.ingestion.document_identity import compute_file_hash
from app.ingestion.document_registry import DocumentRegistry
from app.ingestion.pipeline import IngestionResult

@pytest.fixture
def client():
    return TestClient(main.app)


def test_query_endpoint_returns_grounded_answer(client, monkeypatch):
    """Test successful retrieval and answer generation."""

    document = Document(
        page_content="RAG combines retrieval with generation.",
        metadata={
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "research.pdf",
            "page_number": 2,
        },
    )

    expected_response = AnswerResponse(
        answer="RAG combines retrieval with generation.",
        sources=[
            Source(
                chunk_id="chunk-1",
                document_id="doc-1",
                filename="research.pdf",
                page_number=2,
            )
        ],
    )

    retrieval_calls = []
    generation_calls = []

    def fake_retrieve_documents(query, top_k=None, filters=None):
        retrieval_calls.append({
            "query": query,
            "top_k": top_k,
            "filters": filters,
        })
        return [document]

    def fake_generate_answer(question, documents):
        generation_calls.append({
            "question": question,
            "documents": documents,
        })
        return expected_response

    monkeypatch.setattr(
        main,
        "retrieve_documents",
        fake_retrieve_documents,
    )
    monkeypatch.setattr(
        main,
        "generate_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is RAG?",
            "top_k": 4,
            "filters": {"filename": "research.pdf"},
        },
    )

    assert response.status_code == 200
    assert response.json() == expected_response.model_dump()

    assert retrieval_calls == [
        {
            "query": "What is RAG?",
            "top_k": 4,
            "filters": {"filename": "research.pdf"},
        }
    ]

    assert generation_calls[0]["question"] == "What is RAG?"
    assert generation_calls[0]["documents"] == [document]


def test_query_endpoint_handles_empty_retrieval_results(
    client,
    monkeypatch,
):
    """Test that empty retrieval results reach answer generation."""

    expected_response = AnswerResponse(
        answer=(
            "I could not find relevant information in the available "
            "documents to answer this question."
        ),
        sources=[],
    )

    generation_calls = []

    monkeypatch.setattr(
        main,
        "retrieve_documents",
        lambda **kwargs: [],
    )

    def fake_generate_answer(question, documents):
        generation_calls.append(documents)
        return expected_response

    monkeypatch.setattr(
        main,
        "generate_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200
    assert response.json() == expected_response.model_dump()
    assert generation_calls == [[]]


def test_query_endpoint_rejects_whitespace_question(client):
    response = client.post(
        "/api/v1/query",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_invalid_top_k(client):
    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is RAG?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_unsupported_filter(client):
    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is RAG?",
            "filters": {"invalid_field": "test"},
        },
    )

    assert response.status_code == 400
    assert "Unsupported filter field" in response.json()["detail"]


def test_query_endpoint_returns_400_for_retrieval_validation_error(
    client,
    monkeypatch,
):
    """Test translation of retrieval ValueError into HTTP 400."""

    def fake_retrieve_documents(**kwargs):
        raise ValueError("Invalid retrieval parameters.")

    monkeypatch.setattr(
        main,
        "retrieve_documents",
        fake_retrieve_documents,
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid retrieval parameters."


def test_query_endpoint_handles_unexpected_retrieval_error(
    client,
    monkeypatch,
    caplog,
):
    """Test safe handling and logging of unexpected retrieval failures."""

    def fake_retrieve_documents(**kwargs):
        raise RuntimeError("Internal database connection details")

    monkeypatch.setattr(
        main,
        "retrieve_documents",
        fake_retrieve_documents,
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An internal error occurred during document retrieval."
    )

    assert "Internal database connection details" not in response.text
    assert "Unexpected error during document retrieval." in caplog.text


def test_query_endpoint_handles_generation_error(
    client,
    monkeypatch,
    caplog,
):
    """Test safe handling of unexpected generation failures."""

    monkeypatch.setattr(
        main,
        "retrieve_documents",
        lambda **kwargs: [],
    )

    def fake_generate_answer(**kwargs):
        raise RuntimeError("Sensitive provider error details")

    monkeypatch.setattr(
        main,
        "generate_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An internal error occurred during answer generation."
    )

    assert "Sensitive provider error details" not in response.text
    assert "Unexpected error during answer generation." in caplog.text


def test_upload_endpoint_successfully_indexes_document(
    client, monkeypatch, tmp_path
):
    """Test successful upload, ingestion, indexing, and registration."""

    monkeypatch.setattr(main.settings, "documents_dir", tmp_path)
    monkeypatch.setattr(main, "document_registry", DocumentRegistry())

    def fake_ingest(file_path, registry, register=True):
        document_id = compute_file_hash(file_path)

        assert register is False

        return IngestionResult(
            document_id=document_id,
            documents=[
                Document(
                    page_content="Research content",
                    metadata={
                        "chunk_id": "chunk-1",
                        "document_id": document_id,
                        "filename": Path(file_path).name,
                    },
                )
            ],
            skipped=False,
        )

    indexed_chunks = []

    def fake_index(chunks):
        indexed_chunks.extend(chunks)
        return ["chunk-1"]

    monkeypatch.setattr(main, "ingest_document", fake_ingest)
    monkeypatch.setattr(main, "index_chunks", fake_index)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "research.pdf",
                b"Research content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["filename"] == "research.pdf"
    assert data["chunk_count"] == 1
    assert data["skipped"] is False
    assert data["document_id"]

    assert indexed_chunks[0].metadata["filename"] == "research.pdf"

    stored_file = tmp_path / f"{data['document_id']}.pdf"

    assert stored_file.exists()
    assert main.document_registry.is_registered(data["document_id"])


def test_upload_endpoint_skips_duplicate_document(
    client, monkeypatch, tmp_path
):
    """Test that identical uploads are not indexed twice."""

    monkeypatch.setattr(main.settings, "documents_dir", tmp_path)
    monkeypatch.setattr(main, "document_registry", DocumentRegistry())

    def fake_ingest(file_path, registry, register=True):
        document_id = compute_file_hash(file_path)

        return IngestionResult(
            document_id=document_id,
            documents=[
                Document(
                    page_content="Research content",
                    metadata={
                        "chunk_id": "chunk-1",
                        "document_id": document_id,
                    },
                )
            ],
            skipped=False,
        )

    index_calls = []

    def fake_index(chunks):
        index_calls.append(chunks)
        return ["chunk-1"]

    monkeypatch.setattr(main, "ingest_document", fake_ingest)
    monkeypatch.setattr(main, "index_chunks", fake_index)

    first_response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("research.txt", b"Research content", "text/plain")},
    )

    second_response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("research.txt", b"Research content", "text/plain")},
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 200

    assert second_response.json()["skipped"] is True
    assert second_response.json()["chunk_count"] == 0

    assert len(index_calls) == 1


def test_upload_endpoint_rejects_unsupported_extension(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("research.docx", b"content", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Only PDF and TXT files are supported." in response.json()["detail"]


def test_upload_endpoint_rejects_empty_file(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(main.settings, "documents_dir", tmp_path)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("research.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert "Uploaded file is empty." in response.json()["detail"]

    assert list(tmp_path.iterdir()) == []


def test_upload_endpoint_rejects_oversized_file(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(main.settings, "documents_dir", tmp_path)
    monkeypatch.setattr(main.settings, "max_upload_size_bytes", 4)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("research.txt", b"12345", "text/plain")},
    )

    assert response.status_code == 413
    assert "maximum allowed size" in response.json()["detail"]

    assert list(tmp_path.iterdir()) == []


def test_upload_endpoint_does_not_register_after_indexing_failure(
    client, monkeypatch, tmp_path
):
    """Test cleanup and deferred registration when indexing fails."""

    monkeypatch.setattr(main.settings, "documents_dir", tmp_path)
    monkeypatch.setattr(main, "document_registry", DocumentRegistry())

    def fake_ingest(file_path, registry, register=True):
        document_id = compute_file_hash(file_path)

        return IngestionResult(
            document_id=document_id,
            documents=[
                Document(
                    page_content="Research content",
                    metadata={
                        "chunk_id": "chunk-1",
                        "document_id": document_id,
                    },
                )
            ],
            skipped=False,
        )

    def fake_index(chunks):
        raise RuntimeError("Simulated indexing failure")

    monkeypatch.setattr(main, "ingest_document", fake_ingest)
    monkeypatch.setattr(main, "index_chunks", fake_index)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("research.txt", b"Research content", "text/plain")},
    )

    assert response.status_code == 500
    assert "internal error" in response.json()["detail"].lower()

    assert list(tmp_path.iterdir()) == []
    assert not main.document_registry.is_registered(
        hashlib.sha256(b"Research content").hexdigest()
    )

def test_upload_endpoint_rejects_malformed_pdf(
    client, monkeypatch, tmp_path
):
    """Test malformed PDFs are rejected and uploaded files are cleaned up."""

    monkeypatch.setattr(main.settings, "documents_dir", tmp_path)
    monkeypatch.setattr(main, "document_registry", DocumentRegistry())

    content = b"This is not a valid PDF."
    document_id = hashlib.sha256(content).hexdigest()

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "malformed.pdf",
                content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]

    # The failed upload must not leave a stored file behind.
    assert list(tmp_path.iterdir()) == []

    # A failed ingestion must not register the document.
    assert not main.document_registry.is_registered(document_id)

def test_upload_endpoint_rejects_document_with_no_extractable_text(
    client, monkeypatch, tmp_path
):
    """Test that documents with no extracted text are not indexed."""

    monkeypatch.setattr(main.settings, "documents_dir", tmp_path)
    monkeypatch.setattr(main, "document_registry", DocumentRegistry())

    def fake_ingest(file_path, registry, register=True):
        return IngestionResult(
            document_id=compute_file_hash(file_path),
            documents=[],
            skipped=False,
        )

    monkeypatch.setattr(main, "ingest_document", fake_ingest)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "empty-content.txt",
                b"Some file content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422
    assert "No extractable text" in response.json()["detail"]

    # The failed upload must be cleaned up.
    assert list(tmp_path.iterdir()) == []

    # The document must not be registered.
    document_id = hashlib.sha256(b"Some file content").hexdigest()
    assert not main.document_registry.is_registered(document_id)