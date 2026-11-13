
from app.ingestion.document_registry import DocumentRegistry


def test_new_document_is_registered():
    registry = DocumentRegistry()

    result = registry.register("document_123")

    assert result is True
    assert registry.is_registered("document_123") is True


def test_duplicate_document_is_not_registered_again():
    registry = DocumentRegistry()

    first_result = registry.register("document_123")
    second_result = registry.register("document_123")

    assert first_result is True
    assert second_result is False


def test_different_documents_are_registered_independently():
    registry = DocumentRegistry()

    assert registry.register("document_123") is True
    assert registry.register("document_456") is True

    assert registry.is_registered("document_123") is True
    assert registry.is_registered("document_456") is True


def test_unregistered_document_returns_false():
    registry = DocumentRegistry()

    assert registry.is_registered("unknown_document") is False


def test_separate_registries_have_independent_state():
    first_registry = DocumentRegistry()
    second_registry = DocumentRegistry()

    first_registry.register("document_123")

    assert first_registry.is_registered("document_123") is True
    assert second_registry.is_registered("document_123") is False
