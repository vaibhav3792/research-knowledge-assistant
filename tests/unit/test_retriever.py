
import pytest
from langchain_core.documents import Document

from app.retrieval import retriever


class FakeVectorStore:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.last_query = None
        self.last_k = None
        self.last_filter = None

    def similarity_search(self, query, k, filter=None):
        self.last_query = query
        self.last_k = k
        self.last_filter = filter

        if self.error:
            raise self.error

        return self.results


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Research on retrieval augmented generation.",
            metadata={
                "document_id": "doc-123",
                "filename": "research.pdf",
                "page_number": 2,
                "chunk_id": "chunk-001",
            },
        ),
        Document(
            page_content="Research on vector databases.",
            metadata={
                "document_id": "doc-123",
                "filename": "research.pdf",
                "page_number": 3,
                "chunk_id": "chunk-002",
            },
        ),
    ]


@pytest.fixture
def fake_store(monkeypatch):
    store = FakeVectorStore()

    monkeypatch.setattr(
        retriever,
        "get_vector_store",
        lambda: store,
    )

    return store


def test_retrieve_documents_uses_default_top_k(
    monkeypatch, sample_documents
):
    fake_store = FakeVectorStore(results=sample_documents)

    monkeypatch.setattr(
        retriever,
        "get_vector_store",
        lambda: fake_store,
    )

    monkeypatch.setattr(
        retriever.settings,
        "retrieval_top_k",
        4,
    )

    results = retriever.retrieve_documents("What is RAG?")

    assert results == sample_documents
    assert fake_store.last_query == "What is RAG?"
    assert fake_store.last_k == 4
    assert fake_store.last_filter is None


def test_retrieve_documents_uses_explicit_top_k(
    monkeypatch, sample_documents
):
    fake_store = FakeVectorStore(results=sample_documents)

    monkeypatch.setattr(
        retriever,
        "get_vector_store",
        lambda: fake_store,
    )

    results = retriever.retrieve_documents(
        query="Explain vector databases",
        top_k=2,
    )

    assert results == sample_documents
    assert fake_store.last_k == 2


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "\n\t",
        None,
        123,
    ],
)
def test_retrieve_documents_rejects_invalid_query(query):
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve_documents(query)


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        1.5,
        "4",
        True,
        False,
    ],
)
def test_retrieve_documents_rejects_invalid_top_k(
    fake_store, top_k
):
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve_documents(
            query="What is RAG?",
            top_k=top_k,
        )


def test_retrieve_documents_returns_empty_list_when_no_matches(
    fake_store,
):
    results = retriever.retrieve_documents("Unknown topic")

    assert results == []


def test_retrieve_documents_preserves_document_metadata(
    monkeypatch, sample_documents
):
    fake_store = FakeVectorStore(results=sample_documents)

    monkeypatch.setattr(
        retriever,
        "get_vector_store",
        lambda: fake_store,
    )

    results = retriever.retrieve_documents("What is RAG?")

    assert results[0].metadata["document_id"] == "doc-123"
    assert results[0].metadata["filename"] == "research.pdf"
    assert results[0].metadata["page_number"] == 2
    assert results[0].metadata["chunk_id"] == "chunk-001"


def test_retrieve_documents_propagates_vector_store_errors(
    monkeypatch,
):
    fake_store = FakeVectorStore(
        error=RuntimeError("Vector database unavailable")
    )

    monkeypatch.setattr(
        retriever,
        "get_vector_store",
        lambda: fake_store,
    )

    with pytest.raises(
        RuntimeError,
        match="Vector database unavailable",
    ):
        retriever.retrieve_documents("What is RAG?")


# Metadata filtering tests


@pytest.mark.parametrize(
    "filters",
    [
        {"document_id": "doc-123"},
        {"filename": "research.pdf"},
        {"page_number": 2},
    ],
)
def test_retrieve_documents_passes_single_metadata_filter(
    fake_store, filters
):
    retriever.retrieve_documents(
        query="What is RAG?",
        filters=filters,
    )

    assert fake_store.last_filter == filters


def test_retrieve_documents_combines_multiple_filters_with_and(
    fake_store,
):
    retriever.retrieve_documents(
        query="What is RAG?",
        filters={
            "document_id": "doc-123",
            "page_number": 2,
        },
    )

    assert fake_store.last_filter == {
        "$and": [
            {"document_id": "doc-123"},
            {"page_number": 2},
        ]
    }


def test_retrieve_documents_supports_all_three_filters(
    fake_store,
):
    retriever.retrieve_documents(
        query="What is RAG?",
        filters={
            "document_id": "doc-123",
            "filename": "research.pdf",
            "page_number": 2,
        },
    )

    assert fake_store.last_filter == {
        "$and": [
            {"document_id": "doc-123"},
            {"filename": "research.pdf"},
            {"page_number": 2},
        ]
    }


@pytest.mark.parametrize(
    "filters",
    [
        None,
        {},
    ],
)
def test_retrieve_documents_without_filters(
    fake_store, filters
):
    retriever.retrieve_documents(
        query="What is RAG?",
        filters=filters,
    )

    assert fake_store.last_filter is None


@pytest.mark.parametrize(
    "filters",
    [
        "document_id",
        ["doc-123"],
        123,
    ],
)
def test_retrieve_documents_rejects_non_dictionary_filters(
    fake_store, filters
):
    with pytest.raises(ValueError, match="filters"):
        retriever.retrieve_documents(
            query="What is RAG?",
            filters=filters,
        )


def test_retrieve_documents_rejects_unsupported_filter_field(
    fake_store,
):
    with pytest.raises(ValueError, match="Unsupported filter field"):
        retriever.retrieve_documents(
            query="What is RAG?",
            filters={"chunk_id": "chunk-001"},
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_id", ""),
        ("document_id", "   "),
        ("document_id", None),
        ("document_id", 123),
        ("filename", ""),
        ("filename", "   "),
        ("filename", None),
        ("filename", 123),
    ],
)
def test_retrieve_documents_rejects_invalid_string_filters(
    fake_store, field, value
):
    with pytest.raises(ValueError):
        retriever.retrieve_documents(
            query="What is RAG?",
            filters={field: value},
        )


@pytest.mark.parametrize(
    "page_number",
    [
        0,
        -1,
        1.5,
        "2",
        True,
        False,
        None,
    ],
)
def test_retrieve_documents_rejects_invalid_page_number(
    fake_store, page_number
):
    with pytest.raises(ValueError, match="page_number"):
        retriever.retrieve_documents(
            query="What is RAG?",
            filters={"page_number": page_number},
        )
