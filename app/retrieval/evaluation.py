
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    """Evaluation metrics for a single retrieval query."""

    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float


def calculate_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> RetrievalMetrics:
    """Calculate Precision@K, Recall@K, and Hit Rate@K.

    Args:
        retrieved_ids: Chunk IDs returned by the retriever, in ranked order.
        relevant_ids: Set of chunk IDs considered relevant to the query.
        k: Maximum number of retrieved results to evaluate.

    Returns:
        RetrievalMetrics containing the three evaluation metrics.

    Raises:
        ValueError: If k is not a positive integer or relevant_ids is empty.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty.")

    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("retrieved_ids must not contain duplicate IDs.")

    top_k_ids = retrieved_ids[:k]

    relevant_retrieved = len(set(top_k_ids) & relevant_ids)

    precision = relevant_retrieved / k
    recall = relevant_retrieved / len(relevant_ids)
    hit_rate = 1.0 if relevant_retrieved > 0 else 0.0

    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        hit_rate_at_k=hit_rate,
    )


def calculate_average_metrics(
    metrics: list[RetrievalMetrics],
) -> RetrievalMetrics:
    """Calculate mean retrieval metrics across multiple queries."""

    if not metrics:
        raise ValueError("metrics must contain at least one result.")

    count = len(metrics)

    return RetrievalMetrics(
        precision_at_k=sum(m.precision_at_k for m in metrics) / count,
        recall_at_k=sum(m.recall_at_k for m in metrics) / count,
        hit_rate_at_k=sum(m.hit_rate_at_k for m in metrics) / count,
    )
