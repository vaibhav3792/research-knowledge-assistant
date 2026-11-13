
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.ingestion.pdf_loader import load_pdf


@pytest.fixture
def create_pdf():
    """Factory fixture for generating temporary PDF files."""

    def _create_pdf(path: Path, page_count: int = 1, password=None):
        writer = PdfWriter()

        for _ in range(page_count):
            writer.add_blank_page(width=72, height=72)

        if password:
            writer.encrypt(password)

        with path.open("wb") as file:
            writer.write(file)

        return path

    return _create_pdf


def test_load_valid_pdf(create_pdf, tmp_path):
    pdf_path = create_pdf(tmp_path / "valid.pdf", page_count=1)

    documents = load_pdf(pdf_path)

    assert len(documents) == 1
    assert isinstance(documents[0].page_content, str)


def test_load_multiple_pages(create_pdf, tmp_path):
    pdf_path = create_pdf(tmp_path / "multiple.pdf", page_count=3)

    documents = load_pdf(pdf_path)

    assert len(documents) == 3
    assert [doc.metadata["page_number"] for doc in documents] == [1, 2, 3]


def test_preserve_empty_pages(create_pdf, tmp_path):
    pdf_path = create_pdf(tmp_path / "empty.pdf", page_count=2)

    documents = load_pdf(pdf_path)

    assert len(documents) == 2
    assert all(doc.page_content == "" for doc in documents)


def test_missing_file_raises_error(tmp_path):
    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf(pdf_path)


def test_invalid_extension_raises_error(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("This is not a PDF.")

    with pytest.raises(ValueError, match="Expected a PDF file"):
        load_pdf(file_path)


def test_directory_raises_error(tmp_path):
    directory = tmp_path / "folder.pdf"
    directory.mkdir()

    with pytest.raises(ValueError, match="Expected a file"):
        load_pdf(directory)


def test_malformed_pdf_raises_error(tmp_path):
    pdf_path = tmp_path / "malformed.pdf"
    pdf_path.write_bytes(b"This is not a valid PDF.")

    with pytest.raises(ValueError, match="Unable to read malformed"):
        load_pdf(pdf_path)


def test_encrypted_pdf_raises_error(create_pdf, tmp_path):
    pdf_path = create_pdf(
        tmp_path / "encrypted.pdf",
        page_count=1,
        password="test-password",
    )

    with pytest.raises(ValueError, match="Password-protected"):
        load_pdf(pdf_path)


def test_metadata_correctness(create_pdf, tmp_path):
    pdf_path = create_pdf(tmp_path / "metadata.pdf", page_count=2)

    documents = load_pdf(pdf_path)

    expected_source = str(pdf_path.resolve())

    for page_number, document in enumerate(documents, start=1):
        assert document.metadata["source"] == expected_source
        assert document.metadata["filename"] == "metadata.pdf"
        assert document.metadata["page_number"] == page_number
