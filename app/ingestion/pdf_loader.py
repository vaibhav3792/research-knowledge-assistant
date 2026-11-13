
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF and return one LangChain Document per page."""

    path = Path(file_path)

    # Validate input.
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    # Parse the PDF.
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise ValueError(
            f"Unable to read malformed or corrupted PDF: {path}"
        ) from exc

    # Password-protected PDFs are not supported currently.
    if reader.is_encrypted:
        raise ValueError(
            f"Password-protected PDFs are not supported: {path}"
        )

    documents = []
    source = str(path.resolve())

    # Extract each page independently.
    for page_index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract text from page {page_index + 1} "
                f"of PDF: {path}"
            ) from exc

        document = Document(
            page_content=text,
            metadata={
                "source": source,
                "page_number": page_index + 1,
                "filename": path.name,
            },
        )

        documents.append(document)

    return documents
