
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.generation import generator
from app.generation.generator import (
    NO_CONTEXT_ANSWER,
    generate_answer,
)
from app.schemas.answer import AnswerResponse


def make_document(
    content="Research findings about machine learning.",
    metadata=None,
):
    if metadata is None:
        metadata = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "research.pdf",
            "page_number": 3,
        }

    return Document(
        page_content=content,
        metadata=metadata,
    )


class FakeLLM:
    def __init__(self, response_content="The paper reports improved accuracy."):
        self.response_content = response_content
        self.received_messages = None
        self.call_count = 0

    def invoke(self, messages):
        self.received_messages = messages
        self.call_count += 1

        return SimpleNamespace(content=self.response_content)


def test_generate_answer_returns_structured_response(monkeypatch):
    fake_llm = FakeLLM()

    monkeypatch.setattr(
        generator,
        "get_llm",
        lambda: fake_llm,
    )

    document = make_document()

    result = generate_answer(
        question="What are the findings?",
        documents=[document],
    )

    assert isinstance(result, AnswerResponse)
    assert result.answer == "The paper reports improved accuracy."
    assert len(result.sources) == 1

    assert result.sources[0].chunk_id == "chunk-1"
    assert result.sources[0].document_id == "doc-1"
    assert result.sources[0].filename == "research.pdf"
    assert result.sources[0].page_number == 3

    assert fake_llm.call_count == 1


def test_generate_answer_passes_question_and_context_to_llm(monkeypatch):
    fake_llm = FakeLLM()

    monkeypatch.setattr(
        generator,
        "get_llm",
        lambda: fake_llm,
    )

    document = make_document(
        content="The proposed method improved accuracy by 12%."
    )

    generate_answer(
        question="What was the accuracy improvement?",
        documents=[document],
    )

    messages = fake_llm.received_messages

    combined_content = "\n".join(
        message.content for message in messages
    )

    assert "What was the accuracy improvement?" in combined_content
    assert "The proposed method improved accuracy by 12%." in combined_content
    assert "chunk-1" in combined_content


def test_generate_answer_does_not_call_llm_without_context(monkeypatch):
    fake_llm = FakeLLM()

    monkeypatch.setattr(
        generator,
        "get_llm",
        lambda: fake_llm,
    )

    result = generate_answer(
        question="What are the findings?",
        documents=[],
    )

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert fake_llm.call_count == 0


def test_generate_answer_excludes_empty_chunks_from_sources(monkeypatch):
    fake_llm = FakeLLM()

    monkeypatch.setattr(
        generator,
        "get_llm",
        lambda: fake_llm,
    )

    valid_document = make_document()

    empty_document = make_document(
        content="   ",
        metadata={
            "chunk_id": "chunk-empty",
            "document_id": "doc-1",
            "filename": "research.pdf",
            "page_number": 4,
        },
    )

    result = generate_answer(
        question="What are the findings?",
        documents=[valid_document, empty_document],
    )

    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "chunk-1"


@pytest.mark.parametrize(
    "question",
    ["", "   ", None, 123],
)
def test_generate_answer_rejects_invalid_question(question):
    with pytest.raises(ValueError, match="non-empty string"):
        generate_answer(question, [])


def test_generate_answer_rejects_missing_chunk_id(monkeypatch):
    fake_llm = FakeLLM()

    monkeypatch.setattr(
        generator,
        "get_llm",
        lambda: fake_llm,
    )

    document = make_document(
        metadata={
            "document_id": "doc-1",
            "filename": "research.pdf",
            "page_number": 2,
        }
    )

    with pytest.raises(ValueError, match="chunk_id"):
        generate_answer("What are the findings?", [document])


def test_generate_answer_rejects_non_string_llm_response(monkeypatch):
    fake_llm = FakeLLM(response_content=["unexpected", "content"])

    monkeypatch.setattr(
        generator,
        "get_llm",
        lambda: fake_llm,
    )

    with pytest.raises(TypeError, match="unexpected response format"):
        generate_answer("What are the findings?", [make_document()])


def test_generate_answer_rejects_empty_llm_response(monkeypatch):
    fake_llm = FakeLLM(response_content="   ")

    monkeypatch.setattr(
        generator,
        "get_llm",
        lambda: fake_llm,
    )

    with pytest.raises(ValueError, match="empty response"):
        generate_answer("What are the findings?", [make_document()])
