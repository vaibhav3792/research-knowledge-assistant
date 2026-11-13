
import hashlib

import pytest

from app.ingestion.document_identity import compute_file_hash


def test_hash_is_deterministic(tmp_path):
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"research paper content")

    first_hash = compute_file_hash(file_path)
    second_hash = compute_file_hash(file_path)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_identical_content_produces_same_hash(tmp_path):
    first_file = tmp_path / "first.pdf"
    second_file = tmp_path / "renamed.pdf"

    content = b"identical document content"
    first_file.write_bytes(content)
    second_file.write_bytes(content)

    assert compute_file_hash(first_file) == compute_file_hash(second_file)


def test_modified_content_produces_different_hash(tmp_path):
    first_file = tmp_path / "first.pdf"
    second_file = tmp_path / "second.pdf"

    first_file.write_bytes(b"original content")
    second_file.write_bytes(b"modified content")

    assert compute_file_hash(first_file) != compute_file_hash(second_file)


def test_empty_file_hash(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_bytes(b"")

    assert compute_file_hash(file_path) == hashlib.sha256(b"").hexdigest()


def test_missing_file_raises_error(tmp_path):
    file_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        compute_file_hash(file_path)


def test_directory_raises_error(tmp_path):
    with pytest.raises(ValueError, match="Expected a file"):
        compute_file_hash(tmp_path)
