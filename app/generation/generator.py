
from langchain_core.documents import Document

from app.core.llm import get_llm
from app.generation.context import format_context
from app.generation.prompt import RAG_PROMPT
from app.schemas.answer import AnswerResponse, Source


NO_CONTEXT_ANSWER = (
    "I could not find relevant information in the available documents "
    "to answer this question."
)


def _build_sources(documents: list[Document]) -> list[Source]:
    """Build source references from retrieved document metadata."""

    sources = []

    for index, document in enumerate(documents):
        metadata = document.metadata

        if not document.page_content.strip():
            continue

        chunk_id = metadata.get("chunk_id")
        document_id = metadata.get("document_id")
        filename = metadata.get("filename", metadata.get("source"))
        page_number = metadata.get("page_number")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Retrieved document at position {index} has no valid chunk_id"
            )

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                f"Retrieved document at position {index} has no valid document_id"
            )

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                f"Retrieved document at position {index} has no valid filename"
            )

        if page_number is not None and (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError(
                f"Retrieved document at position {index} has an invalid page_number"
            )

        sources.append(
            Source(
                chunk_id=chunk_id,
                document_id=document_id,
                filename=filename,
                page_number=page_number,
            )
        )

    return sources


def generate_answer(
    question: str,
    documents: list[Document],
) -> AnswerResponse:
    """Generate an answer grounded in retrieved documents."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    context = format_context(documents)

    if not context:
        return AnswerResponse(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
        )

    sources = _build_sources(documents)

    messages = RAG_PROMPT.format_messages(
        context=context,
        question=question.strip(),
    )

    response = get_llm().invoke(messages)

    if not isinstance(response.content, str):
        raise TypeError("LLM returned an unexpected response format")

    answer = response.content.strip()

    if not answer:
        raise ValueError("LLM returned an empty response")

    return AnswerResponse(
        answer=answer,
        sources=sources,
    )
