from __future__ import annotations

import json
import urllib.request

from rca.config.settings import Settings
from rca.llm.client import ChatMessage, OllamaLLMClient
from rca.llm.factory import get_llm_client


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
        "max_tokens": 512,
        "stream": False,
    }


def test_ollama_llm_client_embed_uses_openai_compatible_embeddings_endpoint(monkeypatch) -> None:
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
        return FakeResponse({"data": [{"embedding": [0.4, 0.5, 0.6]}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OllamaLLMClient(
        base_url="https://api.example.test/v1",
        model="gpt-test",
        embedding_model="text-embedding-3-small",
        api_key="secret-token",
        api_style="openai",
    )

    vectors = client.embed(["alpha"], dimensions=768)

    assert vectors == [[0.4, 0.5, 0.6]]
    assert requests[0]["url"] == "https://api.example.test/v1/embeddings"
    assert requests[0]["headers"]["Authorization"] == "Bearer secret-token"
    assert requests[0]["payload"] == {
        "model": "text-embedding-3-small",
        "input": ["alpha"],
        "dimensions": 768,
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


def test_get_llm_client_uses_ollama_backend_settings() -> None:
    settings = Settings(
        llm_backend="ollama",
        llm_base_url="http://localhost:11434",
        llm_api_key="ollama",
        generation_model="qwen2.5:14b",
        embedding_model="nomic-embed-text",
    )

    client = get_llm_client(settings)

    assert isinstance(client, OllamaLLMClient)
    assert client.base_url == "http://localhost:11434"
    assert client.model == "qwen2.5:14b"
    assert client.embedding_model == "nomic-embed-text"
    assert client._api_style == "ollama"


def test_get_llm_client_uses_openai_compatible_backend_settings() -> None:
    settings = Settings(
        llm_backend="openai_compatible",
        openai_base_url="https://api.example.test/v1",
        openai_api_key="secret-token",
        openai_chat_model="gpt-4o-mini",
        openai_embed_model="text-embedding-3-small",
    )

    client = get_llm_client(settings)

    assert isinstance(client, OllamaLLMClient)
    assert client.base_url == "https://api.example.test/v1"
    assert client.model == "gpt-4o-mini"
    assert client.embedding_model == "text-embedding-3-small"
    assert client.api_key == "secret-token"
    assert client._api_style == "openai"


def test_ollama_payloads_disable_thinking_by_default_and_respect_settings(monkeypatch) -> None:
    from rca.config.settings import Settings
    from rca.llm.client import ChatMessage
    from rca.llm.factory import get_llm_client

    def capture(client):
        sent: list[dict] = []

        def fake_post(path, payload, timeout=120):
            sent.append(payload)
            message = {"role": "assistant", "content": "ok", "tool_calls": []}
            if "chat/completions" in path:
                return {"choices": [{"message": message}]}
            return {"message": message}

        monkeypatch.setattr(client, "_post_json", fake_post)
        client.chat([ChatMessage(role="user", content="hi")])
        client.chat_with_tools([{"role": "user", "content": "hi"}], tools=[])
        return sent

    default = OllamaLLMClient(base_url="http://localhost:11434", model="qwen3.5:9b")
    # Thinking models (e.g. qwen3.5) otherwise put the whole answer in message.thinking.
    assert [payload["think"] for payload in capture(default)] == [False, False]

    thinking = get_llm_client(
        Settings(
            llm_backend="ollama",
            llm_base_url="http://localhost:11434",
            generation_model="qwen3.5:9b",
            llm_think=True,
        )
    )
    assert [payload["think"] for payload in capture(thinking)] == [True, True]

    openai = get_llm_client(
        Settings(
            llm_backend="openai_compatible",
            generation_model="qwen3.5:9b",
            openai_chat_model="gpt-4o-mini",
            llm_think=True,
        )
    )
    assert all("think" not in payload for payload in capture(openai))


def test_http_errors_keep_the_server_message(monkeypatch) -> None:
    import io
    import urllib.error

    import pytest

    def refuse(request, timeout=120):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"\\"gemma3:12b\\" does not support thinking"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    client = OllamaLLMClient(base_url="http://localhost:11434", model="gemma3:12b", think=True)
    with pytest.raises(RuntimeError) as excinfo:
        client.chat_with_tools([{"role": "user", "content": "hi"}], tools=[])
    message = str(excinfo.value)
    assert message.startswith("HTTP Error 400: Bad Request")
    assert "does not support thinking" in message
