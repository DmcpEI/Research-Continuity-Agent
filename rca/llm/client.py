"""Provider-agnostic LLM interface."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from rca.config.settings import get_settings

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

    def __init__(
        self,
        base_url: str,
        model: str,
        options: dict[str, Any] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model or get_settings().embedding_model
        self.options = {"temperature": 0}
        if options:
            self.options.update(options)

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": self.options,
        }
        result = self._post_json("/api/chat", payload, timeout=120)
        text = result["message"]["content"]
        return ChatResponse(text=text, raw=result)

    def embed(self, texts: list[str], dimensions: int = 768) -> list[list[float]]:
        del dimensions  # Ollama controls embedding dimensionality based on the selected model.
        vectors: list[list[float]] = []
        for text in texts:
            result = self._post_json(
                "/api/embeddings",
                {"model": self.embedding_model, "prompt": text},
                timeout=120,
            )
            embedding = result.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("Ollama embedding response did not include an 'embedding' vector.")
            vectors.append([float(value) for value in embedding])
        return vectors

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        import urllib.request

        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
