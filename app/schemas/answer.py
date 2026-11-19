
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Reference to a retrieved document chunk."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)


class AnswerResponse(BaseModel):
    """Structured response containing an answer and its sources."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
