from __future__ import annotations

import json
import urllib.request

import pytest

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


def test_ollama_payloads_send_keep_alive_from_settings(monkeypatch) -> None:
    # Ollama otherwise keeps every model in RAM for 5 minutes after each request.
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
            if "embed" in path:
                return {"embedding": [0.1], "data": [{"embedding": [0.1]}]}
            return {"message": message}

        monkeypatch.setattr(client, "_post_json", fake_post)
        client.chat([ChatMessage(role="user", content="hi")])
        client.chat_with_tools([{"role": "user", "content": "hi"}], tools=[])
        client.embed(["hi"])
        return sent

    default = get_llm_client(Settings(llm_backend="ollama", generation_model="m"))
    assert [payload["keep_alive"] for payload in capture(default)] == ["1m", "1m", "1m"]

    seconds = get_llm_client(
        Settings(llm_backend="ollama", generation_model="m", llm_keep_alive="60")
    )
    # A unit-less string is an HTTP 400 in Ollama; whole seconds must go out as a number.
    assert [payload["keep_alive"] for payload in capture(seconds)] == [60, 60, 60]

    bare = OllamaLLMClient(base_url="http://localhost:11434", model="m", embedding_model="e")
    assert all("keep_alive" not in payload for payload in capture(bare))

    openai = get_llm_client(
        Settings(llm_backend="openai_compatible", generation_model="m", llm_keep_alive="5m")
    )
    assert all("keep_alive" not in payload for payload in capture(openai))


def test_keep_alive_setting_rejects_values_ollama_cannot_parse(monkeypatch) -> None:
    import pytest

    from rca.config.settings import Settings

    assert Settings(llm_keep_alive="-1").llm_keep_alive == -1
    assert Settings(llm_keep_alive=" 1h30m ").llm_keep_alive == "1h30m"
    assert Settings(llm_keep_alive=0).llm_keep_alive == 0
    monkeypatch.setenv("RCA_LLM_KEEP_ALIVE", "45")
    assert Settings().llm_keep_alive == 45
    # Python's \d also matches non-ASCII digits, which Ollama rejects.
    for bad in ["abc", "", "5 m", "1d", "\u0663s", "\uff11\uff12s"]:
        with pytest.raises(ValueError):
            Settings(llm_keep_alive=bad)


def test_chat_and_agent_default_to_one_small_model(monkeypatch) -> None:
    from rca.config.settings import Settings

    for name in ("RCA_GENERATION_MODEL", "RCA_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)

    # One shared model means only one chat model is loaded in Ollama at a time.
    assert settings.generation_model == settings.agent_model == "qwen3.5:4b"


@pytest.mark.parametrize("api_style", ["ollama", "openai"])
def test_chat_json_asks_for_the_schema_and_plain_chat_does_not(monkeypatch, api_style) -> None:
    from rca.llm.client import ChatMessage

    client = OllamaLLMClient(base_url="http://localhost:11434", model="m", api_style=api_style)
    sent: list[dict] = []

    def fake_post(path, payload, timeout=120):
        sent.append(payload)
        message = {"role": "assistant", "content": '{"ok": true}'}
        return (
            {"choices": [{"message": message}]}
            if "chat/completions" in path
            else {"message": message}
        )

    monkeypatch.setattr(client, "_post_json", fake_post)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    messages = [ChatMessage(role="user", content="hi")]

    reply = client.chat_json(messages, schema)
    client.chat(messages)

    assert reply.text == '{"ok": true}'
    if api_style == "ollama":
        assert sent[0]["format"] == schema
        # A JSON reply cut short is invalid, so it gets more room than a normal answer.
        assert (sent[0]["options"]["num_predict"], sent[1]["options"]["num_predict"]) == (1024, 512)
    else:
        assert sent[0]["response_format"]["json_schema"]["schema"] == schema
        assert (sent[0]["max_tokens"], sent[1]["max_tokens"]) == (1024, 512)
    assert "format" not in sent[1] and "response_format" not in sent[1]
    assert client.options["num_predict"] == 512


def test_clients_without_structured_output_fall_back_to_plain_chat() -> None:
    from rca.llm.client import ChatMessage, EchoLLMClient

    reply = EchoLLMClient().chat_json([ChatMessage(role="user", content="hi")], {"type": "object"})

    assert reply.text == "[stubbed-llm] hi"


class _NdjsonServer:
    """A local HTTP server that streams Ollama-style NDJSON chunks, one every few milliseconds,
    and records the request and whether the client hung up before the stream ended."""

    def __init__(self, pieces: list[str], delay: float = 0.02) -> None:
        import http.server
        import threading

        self.pieces = pieces
        self.payloads: list[dict] = []
        self.written = 0
        self.hung_up = threading.Event()
        self.done = threading.Event()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                import time

                length = int(self.headers["Content-Length"])
                owner.payloads.append(json.loads(self.rfile.read(length)))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                chunks = [
                    {"message": {"role": "assistant", "content": piece}, "done": False}
                    for piece in owner.pieces
                ]
                chunks.append(
                    {
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "prompt_eval_count": 11,
                        "eval_count": len(owner.pieces),
                    }
                )
                try:
                    for chunk in chunks:
                        self.wfile.write(json.dumps(chunk).encode() + b"\n")
                        self.wfile.flush()
                        owner.written += 1
                        time.sleep(delay)
                except (BrokenPipeError, ConnectionResetError):
                    owner.hung_up.set()
                finally:
                    owner.done.set()

            def log_message(self, *args) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_chat_stream_passes_each_piece_on_and_returns_the_whole_reply() -> None:
    server = _NdjsonServer(["Jam", "Packer ", "packs."])
    try:
        client = OllamaLLMClient(base_url=server.url, model="m")
        pieces: list[str] = []
        reply = client.chat_stream([ChatMessage(role="user", content="hi")], pieces.append)

        assert [piece for piece in pieces if piece] == ["Jam", "Packer ", "packs."]
        assert reply.text == "JamPacker packs."
        assert reply.raw["prompt_eval_count"] == 11 and reply.raw["eval_count"] == 3
        assert reply.raw["message"]["content"] == "JamPacker packs."
        assert server.payloads[0]["stream"] is True
        assert server.payloads[0]["model"] == "m"
    finally:
        server.close()


def test_chat_stream_hangs_up_when_on_token_raises() -> None:
    server = _NdjsonServer([f"w{index} " for index in range(200)])
    try:
        client = OllamaLLMClient(base_url=server.url, model="m")

        class Stop(Exception):
            pass

        seen: list[str] = []

        def on_token(piece: str) -> None:
            seen.append(piece)
            if len(seen) == 2:
                raise Stop

        with pytest.raises(Stop):
            client.chat_stream([ChatMessage(role="user", content="hi")], on_token)

        assert server.done.wait(timeout=10)
        assert server.hung_up.is_set()  # the server saw the connection close
        assert server.written < 200
    finally:
        server.close()


def test_chat_stream_raises_the_error_ollama_sends_mid_stream(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def __iter__(self):
            yield b'{"message": {"role": "assistant", "content": "Ja"}, "done": false}\n'
            yield b'{"error": "model runner has unexpectedly stopped"}\n'

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=120: Response())
    client = OllamaLLMClient(base_url="http://ollama.invalid", model="m")

    with pytest.raises(RuntimeError, match="model runner has unexpectedly stopped"):
        client.chat_stream([ChatMessage(role="user", content="hi")], lambda piece: None)


def test_clients_without_streaming_hand_over_the_whole_reply_once() -> None:
    from rca.llm.client import EchoLLMClient

    pieces: list[str] = []
    reply = EchoLLMClient().chat_stream([ChatMessage(role="user", content="hi")], pieces.append)

    assert pieces == ["[stubbed-llm] hi"] and reply.text == "[stubbed-llm] hi"


def _stream_of(lines: list[bytes]):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def __iter__(self):
            yield from lines

    return lambda request, timeout=120: Response()


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        # The connection dropped before Ollama said it was done: not a whole reply.
        ([b'{"message": {"role": "assistant", "content": "Half"}, "done": false}\n'], "ended"),
        ([b"[1, 2]\n"], "unexpected"),
        ([b"not json\n"], "unexpected"),
    ],
)
def test_chat_stream_rejects_a_stream_that_is_not_a_whole_reply(
    monkeypatch, lines: list[bytes], message: str
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _stream_of(lines))
    client = OllamaLLMClient(base_url="http://ollama.invalid", model="m")

    with pytest.raises(RuntimeError, match=message):
        client.chat_stream([ChatMessage(role="user", content="hi")], lambda piece: None)
