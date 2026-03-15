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
