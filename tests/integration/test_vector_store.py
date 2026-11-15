import gc

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import app.ingestion.indexer as indexer


class FakeEmbeddings(Embeddings):
    """Lightweight deterministic embeddings for integration tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def create_test_store(persist_directory) -> Chroma:
    return Chroma(
        collection_name="test_research_documents",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(persist_directory),
    )


def make_chunk(
    chunk_id: str,
    content: str = "Research content about artificial intelligence.",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc-123",
            "chunk_index": 0,
            "filename": "research.txt",
        },
    )


def test_chunks_are_stored_in_chroma(tmp_path, monkeypatch):
    store = create_test_store(tmp_path)

    monkeypatch.setattr(
        indexer,
        "get_vector_store",
        lambda: store,
    )

    chunks = [
        make_chunk("chunk-1"),
        make_chunk("chunk-2", "Research content about machine learning."),
    ]

    indexed_ids = indexer.index_chunks(chunks)

    assert indexed_ids == ["chunk-1", "chunk-2"]
    assert store._collection.count() == 2

    results = store.similarity_search("artificial intelligence", k=2)

    assert len(results) == 2
    assert all(result.metadata["document_id"] == "doc-123" for result in results)
    assert all(result.metadata["filename"] == "research.txt" for result in results)


def test_reindexing_same_ids_does_not_create_duplicates(
    tmp_path,
    monkeypatch,
):
    store = create_test_store(tmp_path)

    monkeypatch.setattr(
        indexer,
        "get_vector_store",
        lambda: store,
    )

    original_chunk = make_chunk("chunk-1", "Original research content.")
    updated_chunk = make_chunk("chunk-1", "Updated research content.")

    first_result = indexer.index_chunks([original_chunk])
    second_result = indexer.index_chunks([updated_chunk])

    assert first_result == ["chunk-1"]
    assert second_result == ["chunk-1"]

    # The same ID should update the existing record.
    assert store._collection.count() == 1

    results = store.similarity_search("research content", k=1)

    assert len(results) == 1
    assert results[0].page_content == "Updated research content."


def test_chroma_data_persists_after_reopening(tmp_path):
    store = create_test_store(tmp_path)

    store.add_documents(
        documents=[make_chunk("persistent-chunk")],
        ids=["persistent-chunk"],
    )

    assert store._collection.count() == 1

    # Release the first store before opening the database again.
    del store
    gc.collect()

    reopened_store = create_test_store(tmp_path)

    assert reopened_store._collection.count() == 1

    results = reopened_store.similarity_search("research content", k=1)

    assert len(results) == 1
    assert results[0].page_content == (
        "Research content about artificial intelligence."
    )
    assert results[0].metadata["chunk_id"] == "persistent-chunk"
