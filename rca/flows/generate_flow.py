"""Generation is intentionally deferred until retrieval quality is validated."""

from __future__ import annotations


class GenerateFlow:
    """Placeholder for generation once retrieval and citations are solid."""

    def generate_answer(self, query: str) -> str:
        raise NotImplementedError(
            "Generation is intentionally deferred. Validate retrieval and citation quality first."
        )
