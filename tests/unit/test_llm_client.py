from __future__ import annotations

import json
import urllib.request

from rca.llm.client import OllamaLLMClient


def test_ollama_llm_client_embed_uses_embeddings_endpoint(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self) -> "FakeResponse":
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
