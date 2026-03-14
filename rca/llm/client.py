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

class OllamaLLMClient(LLMClient):
    """Local generation via Ollama — free, private, no API key needed."""

    def __init__(self, base_url: str, model: str, options: dict[str, Any] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = {"temperature": 0}
        if options:
            self.options.update(options)

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        import json
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": self.options,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        text = result["message"]["content"]
        return ChatResponse(text=text, raw=result)

    def embed(self, texts: list[str], dimensions: int = 768) -> list[list[float]]:
        raise NotImplementedError("Use VectorStore directly for embeddings")
