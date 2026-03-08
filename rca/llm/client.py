"""Provider-agnostic LLM interface."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)


class LLMClient(ABC):
    """Small interface that providers can implement."""

    @abstractmethod
    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str], dimensions: int = 32) -> list[list[float]]:
        raise NotImplementedError


class EchoLLMClient(LLMClient):
    """Deterministic stub client for local development."""

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        prompt = messages[-1].content if messages else ""
        return ChatResponse(text=f"[stubbed-llm] {prompt}", raw={"message_count": len(messages)})

    def embed(self, texts: list[str], dimensions: int = 32) -> list[list[float]]:
        return [_stable_embedding(text, dimensions) for text in texts]


def _stable_embedding(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = list(digest)
    vector = []
    for index in range(dimensions):
        vector.append(values[index % len(values)] / 255.0)
    return vector
