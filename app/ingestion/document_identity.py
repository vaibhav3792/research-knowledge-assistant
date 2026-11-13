
import hashlib
from pathlib import Path


def compute_file_hash(file_path: str | Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    sha256 = hashlib.sha256()

    # Read in chunks to avoid loading large files into memory.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()
