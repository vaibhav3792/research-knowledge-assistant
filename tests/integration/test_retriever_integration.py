import gc

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.retrieval import retriever


class FakeEmbeddings(Embeddings):
    """Lightweight deterministic embeddings for integration tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def create_test_store(persist_directory) -> Chroma:
    return Chroma(
        collection_name="test_retriever",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(persist_directory),
    )


def make_chunk(
    chunk_id: str,
    document_id: str,
    filename: str,
    page_number: int,
    content: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "filename": filename,
            "page_number": page_number,
            "chunk_index": 0,
        },
    )


@pytest.fixture
def indexed_store(tmp_path, monkeypatch):
    """Create an isolated ChromaDB store containing test documents."""

    store = create_test_store(tmp_path)

    documents = [
        make_chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="research_a.pdf",
            page_number=1,
            content="Research about artificial intelligence.",
        ),
        make_chunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            filename="research_a.pdf",
            page_number=2,
            content="Research about machine learning.",
        ),
        make_chunk(
            chunk_id="chunk-3",
            document_id="doc-2",
            filename="research_b.pdf",
            page_number=1,
            content="Research about neural networks.",
        ),
        make_chunk(
            chunk_id="chunk-4",
            document_id="doc-2",
            filename="research_b.pdf",
            page_number=3,
            content="Research about deep learning.",
        ),
    ]

    store.add_documents(
        documents=documents,
        ids=[document.metadata["chunk_id"] for document in documents],
    )

    monkeypatch.setattr(
        retriever,
        "get_vector_store",
        lambda: store,
    )

    yield store

    # Release the ChromaDB connection before temporary directory cleanup.
    del store
    gc.collect()


def test_retrieve_documents_from_real_chroma(indexed_store):
    results = retriever.retrieve_documents(
        query="Research about artificial intelligence",
        top_k=4,
    )

    assert len(results) == 4
    assert all(isinstance(result, Document) for result in results)


def test_retrieve_documents_filters_by_document_id(indexed_store):
    results = retriever.retrieve_documents(
        query="Research about artificial intelligence",
        top_k=4,
        filters={"document_id": "doc-1"},
    )

    assert len(results) == 2
    assert all(
        result.metadata["document_id"] == "doc-1"
        for result in results
    )


def test_retrieve_documents_filters_by_filename(indexed_store):
    results = retriever.retrieve_documents(
        query="Research about artificial intelligence",
        top_k=4,
        filters={"filename": "research_b.pdf"},
    )

    assert len(results) == 2
    assert all(
        result.metadata["filename"] == "research_b.pdf"
        for result in results
    )


def test_retrieve_documents_filters_by_page_number(indexed_store):
    results = retriever.retrieve_documents(
        query="Research about artificial intelligence",
        top_k=4,
        filters={"page_number": 2},
    )

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "chunk-2"
    assert results[0].metadata["page_number"] == 2


def test_retrieve_documents_applies_multiple_filters(indexed_store):
    results = retriever.retrieve_documents(
        query="Research about artificial intelligence",
        top_k=4,
        filters={
            "document_id": "doc-2",
            "page_number": 3,
        },
    )

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "chunk-4"
    assert results[0].metadata["document_id"] == "doc-2"
    assert results[0].metadata["page_number"] == 3


def test_retrieve_documents_returns_empty_for_unmatched_filter(
    indexed_store,
):
    results = retriever.retrieve_documents(
        query="Research about artificial intelligence",
        top_k=4,
        filters={"document_id": "nonexistent-document"},
    )

    assert results == []



def test_retrieve_documents_preserves_metadata(indexed_store):
    results = retriever.retrieve_documents(
        query="Research about artificial intelligence",
        top_k=4,
        filters={
            "document_id": "doc-1",
            "page_number": 1,
        },
    )

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "chunk-1"
    assert results[0].metadata["filename"] == "research_a.pdf"
    assert results[0].metadata["document_id"] == "doc-1"
    assert results[0].metadata["page_number"] == 1