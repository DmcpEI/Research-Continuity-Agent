"""Cross-encoder reranker wrapping sentence-transformers CrossEncoder."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

if TYPE_CHECKING:
    from rca.flows.retrieve_flow import RetrievalHit

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOGGER = logging.getLogger(__name__)
# After a failed load, wait this long before trying again: an offline Hugging Face lookup can be
# slow, and otherwise every request would pay for it.
RETRY_SECONDS = 60.0


class RerankerRetryLater(RuntimeError):
    """Raised instead of loading again while the last failed load is recent."""


def describe(exc: BaseException) -> str:
    """One short line: the type, and the first line of the message if there is one."""
    lines = str(exc).strip().splitlines()
    text = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    return text if len(text) <= 160 else text[:159] + "…"


class CrossEncoderReranker:
    """Cross-encoder reranker. `error` holds the last load or scoring failure until a success."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.model_name = model_name
        self.error: str | None = None
        self._loaded = None
        self._lock = threading.Lock()  # one load at a time, also on Python 3.12+
        self._clock = clock
        self._retry_at = 0.0

    def load(self) -> bool:
        """Load the model now (at startup) and record a failure instead of raising."""
        try:
            self._model(force=True)
        except Exception as exc:
            self.error = describe(exc)
            LOGGER.warning(
                "Cross-encoder reranker %s failed to load: %s", self.model_name, self.error
            )
            return False
        self.error = None
        return True

    def _model(self, force: bool = False):
        with self._lock:
            if self._loaded is None:
                if not force and self.error is not None and self._clock() < self._retry_at:
                    raise RerankerRetryLater(f"reranker unavailable, retrying later: {self.error}")
                if CrossEncoder is None:
                    raise RuntimeError("sentence-transformers is not installed")
                try:
                    self._loaded = CrossEncoder(self.model_name)
                except Exception:
                    self._retry_at = self._clock() + RETRY_SECONDS
                    raise
            return self._loaded

    def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int | None = None
    ) -> list[RetrievalHit]:
        if not hits:
            return hits

        keep = len(hits) if top_k is None else max(0, min(top_k, len(hits)))
        if keep == 0:
            return []

        pairs = [(query, hit.excerpt or hit.title) for hit in hits]
        try:
            scores = self._model().predict(pairs)
        except RerankerRetryLater:
            raise  # keep the error that caused the wait
        except Exception as exc:
            self.error = describe(exc)
            raise
        self.error = None

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
