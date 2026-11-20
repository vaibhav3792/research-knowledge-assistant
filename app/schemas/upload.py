
from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Response returned after uploading and indexing a document."""

    document_id: str
    filename: str
    chunk_count: int
    skipped: bool
    message: str
