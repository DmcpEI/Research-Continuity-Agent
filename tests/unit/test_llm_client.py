from __future__ import annotations

import json
import urllib.request

from rca.llm.client import ChatMessage, OllamaLLMClient


def test_ollama_llm_client_embed_uses_embeddings_endpoint(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(request: urllib.request.Request, timeout: int = 120):
        requests.append(
            {
                "url": request.full_url,
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return FakeResponse({"embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OllamaLLMClient(
        base_url="http://localhost:11434",
        model="qwen2.5:14b",
        embedding_model="nomic-embed-text",
    )

    vectors = client.embed(["alpha", "beta"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert [request["url"] for request in requests] == [
        "http://localhost:11434/api/embeddings",
        "http://localhost:11434/api/embeddings",
    ]
    assert [request["payload"] for request in requests] == [
        {"model": "nomic-embed-text", "prompt": "alpha"},
        {"model": "nomic-embed-text", "prompt": "beta"},
    ]


def test_llm_client_uses_configured_openai_compatible_base_url(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(request: urllib.request.Request, timeout: int = 120):
        requests.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return FakeResponse({"choices": [{"message": {"content": "Configured backend response"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OllamaLLMClient(
        base_url="https://api.example.test/v1",
        model="gpt-test",
        api_key="secret-token",
    )

    response = client.chat([ChatMessage(role="user", content="Hello")])

    assert response.text == "Configured backend response"
    assert requests[0]["url"] == "https://api.example.test/v1/chat/completions"
    assert requests[0]["headers"]["Authorization"] == "Bearer secret-token"
    assert requests[0]["payload"] == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "stream": False,
    }


def test_ollama_llm_client_chat_with_tools_posts_tools_and_tool_name(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(request: urllib.request.Request, timeout: int = 120):
        requests.append(
            {
                "url": request.full_url,
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return FakeResponse(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": {"query": "bin packing"},
                            }
                        }
                    ],
                }
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OllamaLLMClient(
        base_url="http://localhost:11434",
        model="qwen2.5:14b",
        embedding_model="nomic-embed-text",
    )

    response = client.chat_with_tools(
        messages=[
            {"role": "user", "content": "Search for bin packing"},
            {
                "role": "tool",
                "tool_name": "search_knowledge_base",
                "content": "[src:pdf/demo] Bin Packing paper",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search the KB",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ],
    )

    assert response.tool_calls[0]["function"]["name"] == "search_knowledge_base"
    assert requests[0]["url"] == "http://localhost:11434/api/chat"
    assert requests[0]["payload"]["tools"][0]["function"]["name"] == "search_knowledge_base"
    assert requests[0]["payload"]["messages"][1]["tool_name"] == "search_knowledge_base"
