import pytest
from langchain_core.documents import Document

import app.ingestion.indexer as indexer


def make_chunk(chunk_id: str) -> Document:
    return Document(
        page_content="Research content",
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc-123",
            "chunk_index": 0,
        },
    )


def test_empty_chunks_return_empty_list(monkeypatch):
    def unexpected_vector_store():
        raise AssertionError("Vector store should not be initialized")

    monkeypatch.setattr(
        indexer,
        "get_vector_store",
        unexpected_vector_store,
    )

    assert indexer.index_chunks([]) == []


def test_chunks_are_indexed_with_deterministic_ids(monkeypatch):
    chunks = [
        make_chunk("chunk-1"),
        make_chunk("chunk-2"),
    ]

    class FakeVectorStore:
        def add_documents(self, documents, ids):
            assert documents == chunks
            assert ids == ["chunk-1", "chunk-2"]
            return ids

    monkeypatch.setattr(
        indexer,
        "get_vector_store",
        lambda: FakeVectorStore(),
    )

    result = indexer.index_chunks(chunks)

    assert result == ["chunk-1", "chunk-2"]


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"chunk_id": ""},
        {"chunk_id": None},
        {"chunk_id": 123},
    ],
)
def test_invalid_chunk_id_raises_error(monkeypatch, metadata):
    chunk = Document(
        page_content="Research content",
        metadata=metadata,
    )

    def unexpected_vector_store():
        raise AssertionError("Invalid chunks must not reach the vector store")

    monkeypatch.setattr(
        indexer,
        "get_vector_store",
        unexpected_vector_store,
    )

    with pytest.raises(ValueError, match="valid chunk_id"):
        indexer.index_chunks([chunk])


def test_duplicate_chunk_ids_raise_error(monkeypatch):
    chunks = [
        make_chunk("duplicate-id"),
        make_chunk("duplicate-id"),
    ]

    def unexpected_vector_store():
        raise AssertionError("Duplicate IDs must not reach the vector store")

    monkeypatch.setattr(
        indexer,
        "get_vector_store",
        unexpected_vector_store,
    )

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        indexer.index_chunks(chunks)


def test_vector_store_errors_are_propagated(monkeypatch):
    chunks = [make_chunk("chunk-1")]

    class FailingVectorStore:
        def add_documents(self, documents, ids):
            raise RuntimeError("Simulated indexing failure")

    monkeypatch.setattr(
        indexer,
        "get_vector_store",
        lambda: FailingVectorStore(),
    )

    with pytest.raises(RuntimeError, match="Simulated indexing failure"):
        indexer.index_chunks(chunks)
