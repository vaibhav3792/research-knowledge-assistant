
from pathlib import Path

from langchain_core.documents import Document


def load_txt(file_path: str | Path) -> list[Document]:
    """Load a TXT file and return it as a LangChain Document."""

    path = Path(file_path)

    # Validate input.
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a TXT file, got: {path.suffix}")

    # Extract text using UTF-8 encoding.
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Unable to decode TXT file using UTF-8: {path}"
        ) from exc

    document = Document(
        page_content=text,
        metadata={
            "source": str(path.resolve()),
            "filename": path.name,
        },
    )

    return [document]
