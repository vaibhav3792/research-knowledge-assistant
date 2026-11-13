
from pathlib import Path

from langchain_core.documents import Document

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.txt_loader import load_txt


def load_document(file_path: str | Path) -> list[Document]:
    """Load a supported document format into LangChain Documents."""

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(path)

    if extension == ".txt":
        return load_txt(path)

    raise ValueError(
        f"Unsupported file format: {extension or 'no extension'}"
    )
