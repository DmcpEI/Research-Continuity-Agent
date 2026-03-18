"""Embedding adapters for vector storage."""

from __future__ import annotations

from collections.abc import Sequence

from rca.llm.client import LLMClient


class ConfigurableEmbeddingFunction:
    """Small Chroma-compatible embedding wrapper backed by the RCA LLM client."""

    def __init__(self, client: LLMClient, dimensions: int) -> None:
        self.client = client
        self.dimensions = dimensions

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return self.client.embed(list(input), dimensions=self.dimensions)
