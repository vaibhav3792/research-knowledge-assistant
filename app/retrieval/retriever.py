
from langchain_core.documents import Document

from app.core.config import settings
from app.core.vector_store import get_vector_store


ALLOWED_FILTER_FIELDS = {
    "document_id",
    "filename",
    "page_number",
}


def _build_metadata_filter(
    filters: dict | None,
) -> dict | None:
    """Validate and convert application filters into Chroma metadata filters."""

    if filters is None:
        return None

    if not isinstance(filters, dict):
        raise ValueError("filters must be a dictionary.")

    if not filters:
        return None

    validated_filters = {}

    for field, value in filters.items():
        if field not in ALLOWED_FILTER_FIELDS:
            raise ValueError(f"Unsupported filter field: {field}")

        if field in {"document_id", "filename"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field} filter must be a non-empty string."
                )

        elif field == "page_number":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    "page_number filter must be a positive integer."
                )

        validated_filters[field] = value

    if len(validated_filters) == 1:
        return validated_filters

    return {
        "$and": [
            {field: value}
            for field, value in validated_filters.items()
        ]
    }


def retrieve_documents(
    query: str,
    top_k: int | None = None,
    filters: dict | None = None,
) -> list[Document]:
    """Retrieve the most relevant document chunks for a query.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of chunks to retrieve. Defaults to
            settings.retrieval_top_k.
        filters: Optional metadata filters. Supported fields are
            document_id, filename, and page_number.

    Returns:
        A list of LangChain Documents, including their original metadata.

    Raises:
        ValueError: If the query, top_k, or filters are invalid.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    if top_k is None:
        top_k = settings.retrieval_top_k
    elif isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    metadata_filter = _build_metadata_filter(filters)

    vector_store = get_vector_store()

    search_arguments = {
        "query": query,
        "k": top_k,
    }

    if metadata_filter is not None:
        search_arguments["filter"] = metadata_filter

    return vector_store.similarity_search(**search_arguments)
