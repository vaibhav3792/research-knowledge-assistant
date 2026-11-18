import json

import pytest

from scripts.evaluate_retrieval import (
    create_evaluation_documents,
    load_evaluation_dataset,
)


@pytest.fixture
def valid_dataset():
    return {
        "description": "Test evaluation dataset",
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "text": "Retrieval augmented generation combines retrieval and generation.",
                "metadata": {
                    "document_id": "doc-1",
                    "filename": "rag.txt",
                },
            },
            {
                "chunk_id": "chunk-2",
                "text": "Vector databases support semantic similarity search.",
                "metadata": {
                    "document_id": "doc-2",
                    "filename": "vectors.txt",
                },
            },
        ],
        "queries": [
            {
                "query_id": "q1",
                "query": "How does RAG work?",
                "relevant_chunk_ids": ["chunk-1"],
            }
        ],
    }


def write_dataset(tmp_path, dataset):
    dataset_path = tmp_path / "evaluation.json"
    dataset_path.write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )
    return dataset_path


def test_load_evaluation_dataset_accepts_valid_dataset(
    tmp_path, valid_dataset
):
    dataset_path = write_dataset(tmp_path, valid_dataset)

    result = load_evaluation_dataset(dataset_path)

    assert result == valid_dataset


def test_load_evaluation_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_evaluation_dataset(tmp_path / "missing.json")


def test_load_evaluation_dataset_rejects_invalid_json(tmp_path):
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_evaluation_dataset(dataset_path)


def test_load_evaluation_dataset_rejects_non_object_root(tmp_path):
    dataset_path = write_dataset(tmp_path, [])

    with pytest.raises(ValueError, match="JSON object"):
        load_evaluation_dataset(dataset_path)


@pytest.mark.parametrize("field", ["chunks", "queries"])
def test_load_evaluation_dataset_rejects_empty_or_missing_sections(
    tmp_path, valid_dataset, field
):
    dataset = valid_dataset.copy()
    dataset[field] = []

    with pytest.raises(ValueError, match="non-empty"):
        load_evaluation_dataset(write_dataset(tmp_path, dataset))


def test_load_evaluation_dataset_rejects_non_object_chunk(
    tmp_path, valid_dataset
):
    valid_dataset["chunks"][0] = "invalid"

    with pytest.raises(ValueError, match="Chunk at index 0"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


@pytest.mark.parametrize("chunk_id", ["", "   ", None, 123])
def test_load_evaluation_dataset_rejects_invalid_chunk_id(
    tmp_path, valid_dataset, chunk_id
):
    valid_dataset["chunks"][0]["chunk_id"] = chunk_id

    with pytest.raises(ValueError, match="chunk_id"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


def test_load_evaluation_dataset_rejects_duplicate_chunk_ids(
    tmp_path, valid_dataset
):
    valid_dataset["chunks"][1]["chunk_id"] = "chunk-1"

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


@pytest.mark.parametrize("text", ["", "   ", None, 123])
def test_load_evaluation_dataset_rejects_invalid_chunk_text(
    tmp_path, valid_dataset, text
):
    valid_dataset["chunks"][0]["text"] = text

    with pytest.raises(ValueError, match="non-empty text"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


@pytest.mark.parametrize("metadata", [None, [], "invalid"])
def test_load_evaluation_dataset_rejects_invalid_metadata(
    tmp_path, valid_dataset, metadata
):
    valid_dataset["chunks"][0]["metadata"] = metadata

    with pytest.raises(ValueError, match="metadata as a dictionary"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


def test_load_evaluation_dataset_rejects_non_object_query(
    tmp_path, valid_dataset
):
    valid_dataset["queries"][0] = "invalid"

    with pytest.raises(ValueError, match="Query at index 0"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


@pytest.mark.parametrize("query_id", ["", "   ", None, 123])
def test_load_evaluation_dataset_rejects_invalid_query_id(
    tmp_path, valid_dataset, query_id
):
    valid_dataset["queries"][0]["query_id"] = query_id

    with pytest.raises(ValueError, match="query_id"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


def test_load_evaluation_dataset_rejects_duplicate_query_ids(
    tmp_path, valid_dataset
):
    valid_dataset["queries"].append(
        {
            "query_id": "q1",
            "query": "Another question?",
            "relevant_chunk_ids": ["chunk-2"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate query ID"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


@pytest.mark.parametrize("query", ["", "   ", None, 123])
def test_load_evaluation_dataset_rejects_invalid_query_text(
    tmp_path, valid_dataset, query
):
    valid_dataset["queries"][0]["query"] = query

    with pytest.raises(ValueError, match="non-empty query text"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


def test_load_evaluation_dataset_rejects_empty_relevant_ids(
    tmp_path, valid_dataset
):
    valid_dataset["queries"][0]["relevant_chunk_ids"] = []

    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


def test_load_evaluation_dataset_rejects_invalid_relevant_id(
    tmp_path, valid_dataset
):
    valid_dataset["queries"][0]["relevant_chunk_ids"] = [""]

    with pytest.raises(ValueError, match="invalid relevant chunk ID"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


def test_load_evaluation_dataset_rejects_duplicate_relevant_ids(
    tmp_path, valid_dataset
):
    valid_dataset["queries"][0]["relevant_chunk_ids"] = [
        "chunk-1",
        "chunk-1",
    ]

    with pytest.raises(ValueError, match="duplicate relevant chunk IDs"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


def test_load_evaluation_dataset_rejects_unknown_chunk_references(
    tmp_path, valid_dataset
):
    valid_dataset["queries"][0]["relevant_chunk_ids"] = ["unknown-chunk"]

    with pytest.raises(ValueError, match="unknown chunk IDs"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))


def test_create_evaluation_documents_preserves_metadata(
    valid_dataset,
):
    documents = create_evaluation_documents(valid_dataset["chunks"])

    assert len(documents) == 2
    assert documents[0].page_content == valid_dataset["chunks"][0]["text"]
    assert documents[0].metadata == {
        "document_id": "doc-1",
        "filename": "rag.txt",
        "chunk_id": "chunk-1",
    }

@pytest.mark.parametrize(
    "metadata_value",
    [
        None,
        [],
        ["value"],
        {"nested": "value"},
    ],
)
def test_load_evaluation_dataset_rejects_unsupported_metadata_values(
    tmp_path, valid_dataset, metadata_value
):
    valid_dataset["chunks"][0]["metadata"]["extra"] = metadata_value

    with pytest.raises(ValueError, match="metadata"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))

@pytest.mark.parametrize(
    "metadata_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_load_evaluation_dataset_rejects_non_finite_metadata(
    tmp_path, valid_dataset, metadata_value
):
    valid_dataset["chunks"][0]["metadata"]["score"] = metadata_value

    with pytest.raises(ValueError, match="finite number"):
        load_evaluation_dataset(write_dataset(tmp_path, valid_dataset))