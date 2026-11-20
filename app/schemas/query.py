
from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator


class QueryRequest(BaseModel):
    """Request payload for querying the research knowledge base."""

    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)
    filters: dict[str, StrictStr | StrictInt] | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject questions containing only whitespace."""
        value = value.strip()

        if not value:
            raise ValueError("question must not be empty.")

        return value
