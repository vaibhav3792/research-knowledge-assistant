
from pathlib import Path

import pytest
from langchain_core.documents import Document

import app.ingestion.loader as loader


def test_pdf_delegation(monkeypatch, tmp_path):
    file_path = tmp_path / "research.pdf"
    expected = [Document(page_content="PDF content")]

    def mock_load_pdf(path):
        assert path == file_path
        return expected

    monkeypatch.setattr(loader, "load_pdf", mock_load_pdf)

    result = loader.load_document(file_path)

    assert result is expected


def test_txt_delegation(monkeypatch, tmp_path):
    file_path = tmp_path / "research.txt"
    expected = [Document(page_content="TXT content")]

    def mock_load_txt(path):
        assert path == file_path
        return expected

    monkeypatch.setattr(loader, "load_txt", mock_load_txt)

    result = loader.load_document(file_path)

    assert result is expected


def test_case_insensitive_extension(monkeypatch, tmp_path):
    file_path = tmp_path / "research.PDF"
    expected = [Document(page_content="PDF content")]

    monkeypatch.setattr(loader, "load_pdf", lambda path: expected)

    result = loader.load_document(file_path)

    assert result is expected


def test_unsupported_extension_raises_error(tmp_path):
    file_path = tmp_path / "research.docx"

    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load_document(file_path)
