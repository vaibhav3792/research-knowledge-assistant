
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.ingestion.txt_loader import load_txt


def test_load_valid_txt(tmp_path):
    file_path = tmp_path / "research.txt"
    file_path.write_text("Research content.", encoding="utf-8")

    documents = load_txt(file_path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "Research content."


def test_preserve_multiline_content(tmp_path):
    file_path = tmp_path / "multiline.txt"
    content = "First line.\nSecond line.\nThird line."
    file_path.write_text(content, encoding="utf-8")

    documents = load_txt(file_path)

    assert documents[0].page_content == content


def test_load_empty_txt(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    documents = load_txt(file_path)

    assert len(documents) == 1
    assert documents[0].page_content == ""


def test_missing_file_raises_error(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_txt(file_path)


def test_invalid_extension_raises_error(tmp_path):
    file_path = tmp_path / "research.pdf"
    file_path.write_text("Some content.", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a TXT file"):
        load_txt(file_path)


def test_directory_raises_error(tmp_path):
    directory = tmp_path / "folder.txt"
    directory.mkdir()

    with pytest.raises(ValueError, match="Expected a file"):
        load_txt(directory)


def test_invalid_utf8_raises_error(tmp_path):
    file_path = tmp_path / "invalid.txt"
    file_path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(ValueError, match="Unable to decode TXT"):
        load_txt(file_path)


def test_metadata_correctness(tmp_path):
    file_path = tmp_path / "metadata.txt"
    file_path.write_text("Test content.", encoding="utf-8")

    documents = load_txt(file_path)

    metadata = documents[0].metadata

    assert metadata["source"] == str(file_path.resolve())
    assert metadata["filename"] == "metadata.txt"
    