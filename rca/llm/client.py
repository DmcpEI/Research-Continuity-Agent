"""Provider-agnostic LLM interface."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from rca.config.settings import get_settings


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)


class ToolChatResponse(BaseModel):
    text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class LLMClient(ABC):
    """Small interface that providers can implement."""

    @abstractmethod
    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str], dimensions: int = 32) -> list[list[float]]:
        raise NotImplementedError

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolChatResponse:
        raise NotImplementedError


class EchoLLMClient(LLMClient):
    """Deterministic stub client for local development."""

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        prompt = messages[-1].content if messages else ""
        return ChatResponse(text=f"[stubbed-llm] {prompt}", raw={"message_count": len(messages)})

    def embed(self, texts: list[str], dimensions: int = 32) -> list[list[float]]:
        return [_stable_embedding(text, dimensions) for text in texts]

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolChatResponse:
        del tools
        prompt = ""
        for message in reversed(messages):
            content = message.get("content")
            if isinstance(content, str) and content:
                prompt = content
                break
        return ToolChatResponse(
            text=f"[stubbed-llm] {prompt}",
            tool_calls=[],
            raw={
                "message": {
                    "role": "assistant",
                    "content": f"[stubbed-llm] {prompt}",
                    "tool_calls": [],
                },
                "message_count": len(messages),
            },
        )


def _stable_embedding(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = list(digest)
    vector = []
    for index in range(dimensions):
        vector.append(values[index % len(values)] / 255.0)
    return vector


class OllamaLLMClient(LLMClient):
    """Default Ollama client with optional OpenAI-compatible API support."""

    def __init__(
        self,
        base_url: str,
        model: str,
        options: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        api_style: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model or get_settings().embedding_model
        self.api_key = api_key or "ollama"
        self.api_style = api_style
        self.options = {"temperature": 0}
        if options:
            self.options.update(options)

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        if self._api_style == "openai":
            payload = {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": self.options.get("temperature", 0),
                "stream": False,
            }
            result = self._post_json(self._openai_path("chat/completions"), payload, timeout=120)
            text = self._extract_openai_text(result)
        else:
            payload = {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
                "options": self.options,
            }
            result = self._post_json("/api/chat", payload, timeout=120)
            text = result["message"]["content"]
        return ChatResponse(text=text, raw=result)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolChatResponse:
        normalized_messages = [self._normalize_tool_message(message) for message in messages]

        if self._api_style == "openai":
            payload = {
                "model": self.model,
                "messages": normalized_messages,
                "tools": tools,
                "temperature": self.options.get("temperature", 0),
                "stream": False,
            }
            result = self._post_json(self._openai_path("chat/completions"), payload, timeout=120)
            message = self._extract_openai_message(result)
            return ToolChatResponse(
                text=self._extract_openai_message_text(message),
                tool_calls=self._extract_openai_tool_calls(message),
                raw=result,
            )

        payload = {
            "model": self.model,
            "messages": normalized_messages,
            "tools": tools,
            "stream": False,
            "options": self.options,
        }
        result = self._post_json("/api/chat", payload, timeout=120)
        message = result.get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("Ollama tool-calling response did not include a message object.")
        tool_calls = message.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []
        text = message.get("content")
        return ToolChatResponse(
            text=text if isinstance(text, str) else "",
            tool_calls=tool_calls,
            raw=result,
        )

    def embed(self, texts: list[str], dimensions: int = 768) -> list[list[float]]:
        if self._api_style == "openai":
            payload = {"model": self.embedding_model, "input": texts}
            if dimensions:
                payload["dimensions"] = dimensions
            result = self._post_json(self._openai_path("embeddings"), payload, timeout=120)
            data = result.get("data")
            if not isinstance(data, list):
                raise RuntimeError(
                    "OpenAI-compatible embedding response did not include a 'data' list."
                )
            vectors: list[list[float]] = []
            for item in data:
                embedding = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(embedding, list):
                    raise RuntimeError(
                        "OpenAI-compatible embedding item did not include an 'embedding' vector."
                    )
                vectors.append([float(value) for value in embedding])
            return vectors

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
                raise RuntimeError(
                    "Ollama embedding response did not include an 'embedding' vector."
                )
            vectors.append([float(value) for value in embedding])
        return vectors

    @property
    def _api_style(self) -> str:
        if self.api_style in {"ollama", "openai"}:
            return self.api_style
        parsed = urlparse(self.base_url)
        if parsed.path.rstrip("/").endswith("/v1"):
            return "openai"
        if self.api_key and self.api_key != "ollama":
            return "openai"
        return "ollama"

    def _openai_path(self, suffix: str) -> str:
        parsed = urlparse(self.base_url)
        if parsed.path.rstrip("/").endswith("/v1"):
            return f"/{suffix}"
        return f"/v1/{suffix}"

    @staticmethod
    def _extract_openai_text(result: dict[str, Any]) -> str:
        return OllamaLLMClient._extract_openai_message_text(
            OllamaLLMClient._extract_openai_message(result)
        )

    @staticmethod
    def _extract_openai_message(result: dict[str, Any]) -> dict[str, Any]:
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI-compatible chat response did not include any choices.")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI-compatible chat response did not include a message object.")
        return message

    @staticmethod
    def _extract_openai_message_text(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)
        raise RuntimeError("OpenAI-compatible chat response did not include text content.")

    @staticmethod
    def _extract_openai_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
        tool_calls = message.get("tool_calls", [])
        return tool_calls if isinstance(tool_calls, list) else []

    @staticmethod
    def _normalize_tool_message(message: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(message)
        content = normalized.get("content")
        if content is None:
            normalized["content"] = ""
        elif not isinstance(content, str):
            normalized["content"] = json.dumps(content, sort_keys=True)

        role = normalized.get("role")
        if role == "tool" and not normalized.get("tool_name"):
            raise ValueError("Tool result messages must include 'tool_name'.")
        return normalized

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        import urllib.request

        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "ollama":
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
