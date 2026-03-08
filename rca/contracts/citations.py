"""Citation DTOs and verification inputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rca.contracts.ids import ChunkID, SourceID


class CitationSpan(BaseModel):
    """A single cited span from a source or chunk."""

    source_id: SourceID
    chunk_id: ChunkID | None = None
    locator: str | None = None
    quote: str | None = None


class Citation(BaseModel):
    """A human-readable citation composed of one or more spans."""

    label: str | None = None
    spans: list[CitationSpan] = Field(default_factory=list)

    def render(self) -> str:
        labels = []
        for span in self.spans:
            base = span.chunk_id or span.source_id
            if span.locator:
                base = f"{base}@{span.locator}"
            labels.append(base)
        prefix = f"{self.label}: " if self.label else ""
        return prefix + "; ".join(labels)


class CitationVerificationRequest(BaseModel):
    """Input payload for a future citation verifier."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    strict: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


def render_inline_citations(citations: list[Citation]) -> str:
    """Render citations into a compact inline form."""

    rendered = [citation.render() for citation in citations]
    return " | ".join(rendered)
