
import pytest

from app.ingestion.document_registry import DocumentRegistry
from app.ingestion.pipeline import ingest_document


def test_new_document_is_ingested(tmp_path):
    file_path = tmp_path / "research.txt"
    file_path.write_text("Research paper content", encoding="utf-8")

    registry = DocumentRegistry()

    result = ingest_document(file_path, registry)

    assert result.skipped is False
    assert len(result.documents) == 1
    assert result.document_id
    assert result.documents[0].page_content == "Research paper content"
    assert result.documents[0].metadata["document_id"] == result.document_id


def test_duplicate_document_is_skipped(tmp_path):
    file_path = tmp_path / "research.txt"
    file_path.write_text("Research paper content", encoding="utf-8")

    registry = DocumentRegistry()

    first_result = ingest_document(file_path, registry)
    second_result = ingest_document(file_path, registry)

    assert first_result.skipped is False
    assert second_result.skipped is True
    assert second_result.documents == []
    assert first_result.document_id == second_result.document_id


def test_identical_files_with_different_names_are_duplicates(tmp_path):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "renamed.txt"

    content = "Identical research content"

    first_file.write_text(content, encoding="utf-8")
    second_file.write_text(content, encoding="utf-8")

    registry = DocumentRegistry()

    first_result = ingest_document(first_file, registry)
    second_result = ingest_document(second_file, registry)

    assert first_result.document_id == second_result.document_id
    assert first_result.skipped is False
    assert second_result.skipped is True


def test_modified_document_is_ingested_again(tmp_path):
    file_path = tmp_path / "research.txt"

    registry = DocumentRegistry()

    file_path.write_text("Original content", encoding="utf-8")
    first_result = ingest_document(file_path, registry)

    file_path.write_text("Modified content", encoding="utf-8")
    second_result = ingest_document(file_path, registry)

    assert first_result.document_id != second_result.document_id
    assert first_result.skipped is False
    assert second_result.skipped is False
    assert second_result.documents[0].page_content == "Modified content"


def test_document_identity_is_added_to_all_pages(tmp_path):
    file_path = tmp_path / "research.pdf"

    # Use the existing research PDF if available.
    from pathlib import Path

    source_pdf = Path("data/raw/research_paper.pdf")

    if not source_pdf.exists():
        pytest.skip("Research PDF not available")

    file_path.write_bytes(source_pdf.read_bytes())

    registry = DocumentRegistry()

    result = ingest_document(file_path, registry)

    assert len(result.documents) > 1

    for document in result.documents:
        assert document.metadata["document_id"] == result.document_id


def test_failed_loading_does_not_register_document(
    tmp_path, monkeypatch
):
    import app.ingestion.pipeline as pipeline

    file_path = tmp_path / "research.txt"
    file_path.write_text("Research content", encoding="utf-8")

    registry = DocumentRegistry()

    def mock_load_document(path):
        raise RuntimeError("Simulated loading failure")

    monkeypatch.setattr(
        pipeline,
        "load_document",
        mock_load_document,
    )

    with pytest.raises(RuntimeError, match="Simulated loading failure"):
        ingest_document(file_path, registry)

    from app.ingestion.document_identity import compute_file_hash

    document_id = compute_file_hash(file_path)

    assert registry.is_registered(document_id) is False
