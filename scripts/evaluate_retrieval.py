import argparse
import json
from pathlib import Path
import math

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import settings
from app.core.embeddings import get_embeddings
from app.retrieval.evaluation import (
    calculate_average_metrics,
    calculate_retrieval_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.json"
)



def load_evaluation_dataset(dataset_path: Path) -> dict:
    """Load and validate the retrieval evaluation dataset."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    try:
        with dataset_path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in evaluation dataset: {exc}"
        ) from exc

    if not isinstance(dataset, dict):
        raise ValueError("Evaluation dataset must be a JSON object.")

    chunks = dataset.get("chunks")
    queries = dataset.get("queries")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "Evaluation dataset must contain a non-empty chunks list."
        )

    if not isinstance(queries, list) or not queries:
        raise ValueError(
            "Evaluation dataset must contain a non-empty queries list."
        )

    chunk_ids = set()

    # Validate chunks.
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk at index {index} must be a JSON object."
            )

        chunk_id = chunk.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Chunk at index {index} must have a non-empty string chunk_id."
            )

        if chunk_id in chunk_ids:
            raise ValueError(
                f"Duplicate chunk ID found: {chunk_id}"
            )

        chunk_ids.add(chunk_id)

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Chunk '{chunk_id}' must have non-empty text."
            )

        metadata = chunk.get("metadata")

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Chunk '{chunk_id}' must have metadata as a dictionary."
            )

        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"Chunk '{chunk_id}' metadata value for '{key}' "
                    "must be a string, integer, float, or boolean."
                )

            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"Chunk '{chunk_id}' metadata value for '{key}' "
                    "must be a finite number."
                )

    query_ids = set()

    # Validate queries.
    for index, query_data in enumerate(queries):
        if not isinstance(query_data, dict):
            raise ValueError(
                f"Query at index {index} must be a JSON object."
            )

        query_id = query_data.get("query_id")

        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(
                f"Query at index {index} must have a non-empty string query_id."
            )

        if query_id in query_ids:
            raise ValueError(
                f"Duplicate query ID found: {query_id}"
            )

        query_ids.add(query_id)

        query = query_data.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Query '{query_id}' must have non-empty query text."
            )

        relevant_ids = query_data.get("relevant_chunk_ids")

        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(
                f"Query '{query_id}' must have a non-empty "
                "relevant_chunk_ids list."
            )

        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in relevant_ids
        ):
            raise ValueError(
                f"Query '{query_id}' contains an invalid relevant chunk ID."
            )

        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(
                f"Query '{query_id}' contains duplicate relevant chunk IDs."
            )

        unknown_ids = set(relevant_ids) - chunk_ids

        if unknown_ids:
            raise ValueError(
                f"Query '{query_id}' references unknown chunk IDs: "
                f"{sorted(unknown_ids)}"
            )

    return dataset


def create_evaluation_documents(chunks: list[dict]) -> list[Document]:
    """Convert evaluation chunks into LangChain Documents."""

    documents = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        metadata = dict(chunk["metadata"])
        metadata["chunk_id"] = chunk_id

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )

    return documents


def evaluate_retrieval(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    top_k: int | None = None,
) -> None:
    """Evaluate semantic retrieval using an isolated in-memory Chroma collection."""

    if top_k is None:
        top_k = settings.retrieval_top_k

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    dataset = load_evaluation_dataset(dataset_path)

    documents = create_evaluation_documents(dataset["chunks"])

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in dataset["chunks"]
    ]

    embeddings = get_embeddings()

    query_metrics = []

    print("\nRetrieval Evaluation")
    print("=" * 60)
    print(f"Dataset: {dataset_path.name}")
    print(f"Total chunks: {len(documents)}")
    print(f"Total queries: {len(dataset['queries'])}")
    print(f"Evaluation K: {top_k}")
    print("=" * 60)

    vector_store = Chroma(
        collection_name="retrieval_evaluation",
        embedding_function=embeddings,
    )

    vector_store.add_documents(
        documents=documents,
        ids=chunk_ids,
    )

    for query_data in dataset["queries"]:
        query_id = query_data["query_id"]
        query = query_data["query"]
        relevant_ids = set(query_data["relevant_chunk_ids"])

        results = vector_store.similarity_search(
            query=query,
            k=top_k,
        )

        retrieved_ids = [
            result.metadata["chunk_id"]
            for result in results
        ]

        metrics = calculate_retrieval_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=top_k,
        )

        query_metrics.append(metrics)

        print(f"\nQuery: {query_id}")
        print(f"Question: {query}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Relevant IDs: {sorted(relevant_ids)}")
        print(
            f"Precision@{top_k}: "
            f"{metrics.precision_at_k:.3f}"
        )
        print(
            f"Recall@{top_k}: "
            f"{metrics.recall_at_k:.3f}"
        )
        print(
            f"Hit Rate@{top_k}: "
            f"{metrics.hit_rate_at_k:.3f}"
        )

    average_metrics = calculate_average_metrics(query_metrics)

    print("\n" + "=" * 60)
    print("AGGREGATE RETRIEVAL RESULTS")
    print("=" * 60)
    print(
        f"Average Precision@{top_k}: "
        f"{average_metrics.precision_at_k:.3f}"
    )
    print(
        f"Average Recall@{top_k}: "
        f"{average_metrics.recall_at_k:.3f}"
    )
    print(
        f"Average Hit Rate@{top_k}: "
        f"{average_metrics.hit_rate_at_k:.3f}"
    )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic retrieval quality."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.retrieval_top_k,
        help="Number of chunks to retrieve per query.",
    )

    args = parser.parse_args()

    evaluate_retrieval(top_k=args.top_k)


if __name__ == "__main__":
    main()
