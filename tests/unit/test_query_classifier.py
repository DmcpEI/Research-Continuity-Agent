from __future__ import annotations

import pytest

from rca.retrieval.query_classifier import QueryType, classify_query


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What does YOLO do?", QueryType.proper_noun),
        ("Results in PointNet++ by Qi et al.", QueryType.proper_noun),
        ("Compare GPT-4 and LLaMA 2023", QueryType.proper_noun),
        ("What is DBLF in JamPacker?", QueryType.proper_noun),
        ("What approaches exist for robotic bin packing?", QueryType.conceptual),
        ("how does reinforcement learning work", QueryType.conceptual),
        ("explain transformer attention mechanisms", QueryType.conceptual),
        ("What did Smith propose?", QueryType.hybrid),
        ("explain Bagging", QueryType.hybrid),
    ],
)
def test_classify_query(query: str, expected: QueryType) -> None:
    assert classify_query(query) == expected
