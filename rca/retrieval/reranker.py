"""Cross-encoder reranker wrapping sentence-transformers CrossEncoder."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

if TYPE_CHECKING:
    from rca.flows.retrieve_flow import RetrievalHit

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder reranker."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    @cached_property
    def _model(self):
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers is not installed")
        return CrossEncoder(self.model_name)

    def rerank(self, query: str, hits: list["RetrievalHit"], top_k: int | None = None) -> list["RetrievalHit"]:
        if not hits:
            return hits

        keep = len(hits) if top_k is None else max(0, min(top_k, len(hits)))
        if keep == 0:
            return []

        pairs = [(query, hit.excerpt or hit.title) for hit in hits]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            hit.model_copy(
                update={
                    "metadata": {**hit.metadata, "rerank_score": float(score)},
                }
            )
            for hit, score in ranked[:keep]
        ]
