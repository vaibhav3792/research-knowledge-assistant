
import pytest

from app.retrieval.evaluation import (
    RetrievalMetrics,
    calculate_average_metrics,
    calculate_retrieval_metrics,
)


def test_calculate_retrieval_metrics():
    retrieved_ids = ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    relevant_ids = {"chunk-1", "chunk-3", "chunk-5"}

    result = calculate_retrieval_metrics(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k=4,
    )

    assert result == RetrievalMetrics(
        precision_at_k=0.5,
        recall_at_k=pytest.approx(2 / 3),
        hit_rate_at_k=1.0,
    )


def test_calculate_retrieval_metrics_respects_k():
    retrieved_ids = ["chunk-1", "chunk-2", "chunk-3"]
    relevant_ids = {"chunk-1", "chunk-3"}

    result = calculate_retrieval_metrics(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k=2,
    )

    assert result.precision_at_k == 0.5
    assert result.recall_at_k == 0.5
    assert result.hit_rate_at_k == 1.0


def test_calculate_retrieval_metrics_with_no_relevant_results():
    retrieved_ids = ["chunk-1", "chunk-2"]
    relevant_ids = {"chunk-3", "chunk-4"}

    result = calculate_retrieval_metrics(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k=2,
    )

    assert result.precision_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.hit_rate_at_k == 0.0


def test_calculate_retrieval_metrics_with_empty_retrieval():
    result = calculate_retrieval_metrics(
        retrieved_ids=[],
        relevant_ids={"chunk-1"},
        k=4,
    )

    assert result.precision_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.hit_rate_at_k == 0.0


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "4", None])
def test_calculate_retrieval_metrics_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        calculate_retrieval_metrics(
            retrieved_ids=["chunk-1"],
            relevant_ids={"chunk-1"},
            k=k,
        )


def test_calculate_retrieval_metrics_rejects_empty_relevant_ids():
    with pytest.raises(ValueError, match="relevant_ids must not be empty"):
        calculate_retrieval_metrics(
            retrieved_ids=["chunk-1"],
            relevant_ids=set(),
            k=1,
        )


def test_calculate_retrieval_metrics_rejects_duplicate_retrieved_ids():
    with pytest.raises(ValueError, match="duplicate IDs"):
        calculate_retrieval_metrics(
            retrieved_ids=["chunk-1", "chunk-1"],
            relevant_ids={"chunk-1"},
            k=2,
        )


def test_calculate_average_metrics():
    metrics = [
        RetrievalMetrics(
            precision_at_k=0.5,
            recall_at_k=0.5,
            hit_rate_at_k=1.0,
        ),
        RetrievalMetrics(
            precision_at_k=0.25,
            recall_at_k=0.5,
            hit_rate_at_k=1.0,
        ),
    ]

    result = calculate_average_metrics(metrics)

    assert result.precision_at_k == pytest.approx(0.375)
    assert result.recall_at_k == pytest.approx(0.5)
    assert result.hit_rate_at_k == pytest.approx(1.0)


def test_calculate_average_metrics_rejects_empty_list():
    with pytest.raises(ValueError, match="metrics must contain"):
        calculate_average_metrics([])
