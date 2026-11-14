from langchain_core.documents import Document

from app.core.config import settings
from app.ingestion.chunker import chunk_documents


def test_short_document_produces_one_chunk():
    document = Document(
        page_content="Short research content",
        metadata={"document_id": "doc-123"},
    )

    chunks = chunk_documents([document])

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short research content"


def test_long_document_produces_multiple_chunks():
    document = Document(
        page_content="Research content. " * 200,
        metadata={"document_id": "doc-123"},
    )

    chunks = chunk_documents([document])

    assert len(chunks) > 1


def test_chunk_size_respects_configuration():
    document = Document(
        page_content="Research content. " * 200,
        metadata={"document_id": "doc-123"},
    )

    chunks = chunk_documents([document])

    assert all(
        len(chunk.page_content) <= settings.chunk_size
        for chunk in chunks
    )


def test_chunk_overlap_is_applied():
    document = Document(
        page_content="A" * 1500,
        metadata={"document_id": "doc-123"},
    )

    chunks = chunk_documents([document])

    assert len(chunks) > 1
    assert (
        chunks[0].page_content[-settings.chunk_overlap:]
        ==
        chunks[1].page_content[:settings.chunk_overlap]
    )


def test_original_metadata_is_preserved():
    metadata = {
        "document_id": "doc-123",
        "filename": "research.pdf",
        "source": "/documents/research.pdf",
        "page_number": 3,
        "custom_field": "custom_value",
    }

    document = Document(
        page_content="Research content",
        metadata=metadata,
    )

    chunks = chunk_documents([document])

    assert len(chunks) == 1

    for key, value in metadata.items():
        assert chunks[0].metadata[key] == value


def test_pdf_page_number_is_preserved():
    document = Document(
        page_content="PDF page content",
        metadata={
            "document_id": "doc-123",
            "page_number": 5,
        },
    )

    chunks = chunk_documents([document])

    assert len(chunks) == 1
    assert chunks[0].metadata["page_number"] == 5


def test_txt_document_does_not_receive_page_number():
    document = Document(
        page_content="TXT document content",
        metadata={
            "document_id": "doc-123",
            "filename": "research.txt",
        },
    )

    chunks = chunk_documents([document])

    assert len(chunks) == 1
    assert "page_number" not in chunks[0].metadata


def test_document_id_propagates_to_all_chunks():
    document = Document(
        page_content="Research content. " * 200,
        metadata={"document_id": "doc-123"},
    )

    chunks = chunk_documents([document])

    assert len(chunks) > 1

    assert all(
        chunk.metadata["document_id"] == "doc-123"
        for chunk in chunks
    )


def test_chunk_ids_are_deterministic():
    document = Document(
        page_content="Research content. " * 100,
        metadata={"document_id": "doc-123"},
    )

    first_chunks = chunk_documents([document])
    second_chunks = chunk_documents([document])

    first_ids = [
        chunk.metadata["chunk_id"]
        for chunk in first_chunks
    ]

    second_ids = [
        chunk.metadata["chunk_id"]
        for chunk in second_chunks
    ]

    assert first_ids == second_ids


def test_different_documents_produce_different_chunk_ids():
    first_document = Document(
        page_content="Identical research content",
        metadata={"document_id": "doc-123"},
    )

    second_document = Document(
        page_content="Identical research content",
        metadata={"document_id": "doc-456"},
    )

    first_chunks = chunk_documents([first_document])
    second_chunks = chunk_documents([second_document])

    assert (
        first_chunks[0].metadata["chunk_id"]
        != second_chunks[0].metadata["chunk_id"]
    )


def test_empty_document_produces_no_chunks():
    document = Document(
        page_content="",
        metadata={"document_id": "doc-123"},
    )

    chunks = chunk_documents([document])

    assert chunks == []


def test_whitespace_only_document_produces_no_chunks():
    document = Document(
        page_content="   \n\t   ",
        metadata={"document_id": "doc-123"},
    )

    chunks = chunk_documents([document])

    assert chunks == []


def test_missing_optional_metadata_does_not_break_chunking():
    document = Document(
        page_content="Research content",
        metadata={},
    )

    chunks = chunk_documents([document])

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_id"]
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_indexes_are_consistent():
    document = Document(
        page_content="Research content. " * 200,
        metadata={"document_id": "doc-123"},
    )

    chunks = chunk_documents([document])

    indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
    ]

    assert indexes == list(range(len(chunks)))


def test_chunk_indexes_restart_for_each_source_document():
    first_document = Document(
        page_content="First research content. " * 100,
        metadata={
            "document_id": "doc-123",
            "page_number": 1,
        },
    )

    second_document = Document(
        page_content="Second research content. " * 100,
        metadata={
            "document_id": "doc-123",
            "page_number": 2,
        },
    )

    chunks = chunk_documents([first_document, second_document])

    first_page_indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
        if chunk.metadata["page_number"] == 1
    ]

    second_page_indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
        if chunk.metadata["page_number"] == 2
    ]

    assert first_page_indexes == list(range(len(first_page_indexes)))
    assert second_page_indexes == list(range(len(second_page_indexes)))


def test_exact_chunk_size_document_produces_one_chunk():
    document = Document(
        page_content="A" * settings.chunk_size,
        metadata={"document_id": "doc-123"},
    )

    chunks = chunk_documents([document])

    assert len(chunks) == 1
    assert len(chunks[0].page_content) == settings.chunk_size
