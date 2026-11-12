from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rca.flows.retrieve_flow import RetrievalHit
from rca.retrieval.reranker import CrossEncoderReranker


def _make_hit(node_id: str, score: float, excerpt: str = "") -> RetrievalHit:
    return RetrievalHit(node_id=node_id, score=score, title=node_id, excerpt=excerpt)


@patch("rca.retrieval.reranker.CrossEncoder")
def test_rerank_orders_by_cross_encoder_score(mock_ce_cls) -> None:
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5]
    mock_ce_cls.return_value = mock_model

    reranker = CrossEncoderReranker()
    hits = [
        _make_hit("a", score=0.95),
        _make_hit("b", score=0.80),
        _make_hit("c", score=0.70),
    ]

    result = reranker.rerank("test query", hits, top_k=3)

    assert [hit.node_id for hit in result] == ["b", "c", "a"]
    assert result[0].score == pytest.approx(0.80)
    assert result[0].metadata["rerank_score"] == pytest.approx(0.9)


@patch("rca.retrieval.reranker.CrossEncoder")
def test_rerank_top_k_truncates(mock_ce_cls) -> None:
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.3, 0.8, 0.6, 0.9, 0.1]
    mock_ce_cls.return_value = mock_model

    reranker = CrossEncoderReranker()
    hits = [_make_hit(str(i), score=0.5) for i in range(5)]

    result = reranker.rerank("q", hits, top_k=2)

    assert len(result) == 2
    assert result[0].node_id == "3"
    assert result[1].node_id == "1"
    assert result[0].metadata["rerank_score"] == pytest.approx(0.9)


def test_rerank_empty_input() -> None:
    reranker = CrossEncoderReranker()
    assert reranker.rerank("q", []) == []


@patch("rca.retrieval.reranker.CrossEncoder")
def test_load_reports_a_failure_and_a_later_success_clears_it(mock_ce_cls) -> None:
    mock_ce_cls.side_effect = [OSError("no cached model, offline"), MagicMock()]
    reranker = CrossEncoderReranker()

    assert reranker.load() is False
    assert reranker.error == "OSError: no cached model, offline"
    assert reranker.load() is True
    assert reranker.error is None


@patch("rca.retrieval.reranker.CrossEncoder")
def test_a_failed_rerank_is_recorded_and_raised(mock_ce_cls) -> None:
    mock_model = MagicMock()
    mock_model.predict.side_effect = [RuntimeError("MPS out of memory"), [0.2, 0.8]]
    mock_ce_cls.return_value = mock_model
    reranker = CrossEncoderReranker()
    hits = [_make_hit("a", 0.9), _make_hit("b", 0.8)]

    with pytest.raises(RuntimeError):
        reranker.rerank("q", hits)
    assert reranker.error == "RuntimeError: MPS out of memory"
    reranker.rerank("q", hits)
    assert reranker.error is None


@patch("rca.retrieval.reranker.CrossEncoder")
def test_after_a_failed_load_reranks_retry_at_most_once_a_minute(mock_ce_cls) -> None:
    now = [1000.0]
    mock_ce_cls.side_effect = [OSError("offline"), OSError("offline"), MagicMock()]
    reranker = CrossEncoderReranker(clock=lambda: now[0])
    hits = [_make_hit("a", 0.9)]

    assert reranker.load() is False
    with pytest.raises(RuntimeError, match="retrying"):
        reranker.rerank("q", hits)  # within the minute: no second slow load
    assert mock_ce_cls.call_count == 1
    assert reranker.error == "OSError: offline"

    now[0] += 61
    with pytest.raises(OSError):
        reranker.rerank("q", hits)
    assert mock_ce_cls.call_count == 2
    now[0] += 61
    mock_ce_cls.side_effect = None
    mock_ce_cls.return_value.predict.return_value = [0.5]
    reranker.rerank("q", hits)
    assert reranker.error is None


@pytest.mark.parametrize(
    ("exc", "expected"),
    [(RuntimeError(""), "RuntimeError"), (OSError("x" * 500), "OSError: " + "x" * 150 + "…")],
)
@patch("rca.retrieval.reranker.CrossEncoder")
def test_errors_are_recorded_as_one_short_line(mock_ce_cls, exc, expected) -> None:
    mock_ce_cls.side_effect = exc
    reranker = CrossEncoderReranker()
    reranker.load()
    assert reranker.error == expected
