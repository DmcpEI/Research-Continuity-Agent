from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

from fastapi.testclient import TestClient

from rca.agent.contracts import AgentResult, AgentTrace
from rca.api.main import create_app
from rca.config.settings import Settings
from rca.contracts.nodes import Edge, EdgeKind, Node, NodeKind
from rca.contracts.revisions import IngestStatus, SourceRevision
from rca.flows.generate_flow import Citation, GeneratedAnswer
from rca.flows.ingest_flow import IngestResult
from rca.flows.retrieve_flow import RetrieveFlow
from rca.store.graph_store import GraphStore
from rca.store.vector_store import VectorStore


def _settings_for(tmp_path: Path) -> Settings:
    runtime_dir = tmp_path / ".rca"
    return Settings(
        workspace_root=tmp_path,
        filesystem_root=tmp_path,
        llm_backend="openai_compatible",
        openai_chat_model="gpt-4o-mini",
        data_dir=runtime_dir,
        graph_db_path=runtime_dir / "graph.sqlite3",
        vector_dir=runtime_dir / "vectors",
        telemetry_log_path=runtime_dir / "telemetry.jsonl",
        experiment_db_path=runtime_dir / "experiments.sqlite3",
    )


def test_sources_endpoint_lists_sources_with_counts_and_revision(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)

    source_id = "src:pdf/sample"
    chunk_ids = ["chk:pdf/sample:0000", "chk:pdf/sample:0001"]

    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="Sample Paper",
            metadata={"latest_path": str(tmp_path / "sample.pdf"), "content_sha256": "abc123"},
        )
    )
    for chunk_id in chunk_ids:
        store.upsert_node(Node(id=chunk_id, kind=NodeKind.chunk, title=chunk_id, text="chunk"))
        store.upsert_edge(Edge(source=source_id, target=chunk_id, kind=EdgeKind.contains))

    store.insert_source_revision(
        SourceRevision(
            source_id=source_id,
            revision_number=1,
            ingest_status=IngestStatus.created,
            path=str(tmp_path / "sample.pdf"),
            ingest_name="sample",
            title="Sample Paper",
            content_sha256="abc123",
        )
    )

    client = TestClient(create_app(settings=settings, graph_store=store))
    response = client.get("/sources")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == source_id
    assert payload[0]["chunk_count"] == 2
    assert payload[0]["latest_path"].endswith("sample.pdf")
    assert payload[0]["content_sha256"] == "abc123"
    assert payload[0]["latest_revision_created_at"] is not None


def test_source_detail_endpoint_returns_metadata_and_revisions(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)

    source_id = "src:pdf/sample"
    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="Sample Paper",
            metadata={"latest_path": str(tmp_path / "sample.pdf"), "content_sha256": "abc123"},
        )
    )

    store.insert_source_revision(
        SourceRevision(
            source_id=source_id,
            revision_number=1,
            ingest_status=IngestStatus.created,
            path=str(tmp_path / "sample.pdf"),
            ingest_name="sample",
            title="Sample Paper",
            content_sha256="abc123",
        )
    )

    client = TestClient(create_app(settings=settings, graph_store=store))
    response = client.get(f"/sources/{source_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == source_id
    assert payload["metadata"]["content_sha256"] == "abc123"
    assert len(payload["revisions"]) == 1
    assert payload["revisions"][0]["revision_number"] == 1


def test_source_detail_endpoint_returns_404_when_missing(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)

    client = TestClient(create_app(settings=settings, graph_store=store))
    response = client.get("/sources/src:pdf/missing")

    assert response.status_code == 404


def test_chat_endpoint_returns_generate_flow_payload(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    expected_messages = [
        {"role": "user", "content": "What is RCA?"},
        {"role": "assistant", "content": "RCA is a system."},
    ]

    class FakeGenerateFlow:
        def generate_answer(self, query: str, conversation_messages=None):
            assert query == "what is RCA?"
            assert conversation_messages == expected_messages
            return GeneratedAnswer(
                query=query,
                answer="RCA is a local-first research knowledge system.",
                citations=[
                    Citation(
                        source_id="src:pdf/sample",
                        title="Sample Paper",
                        excerpt="Sample excerpt",
                    )
                ],
                outcome="answered",
                trace=None,
            )

    client = TestClient(
        create_app(
            settings=settings,
            graph_store=store,
            generate_flow=FakeGenerateFlow(),
        )
    )
    response = client.post(
        "/chat",
        json={
            "query": "what is RCA?",
            "conversation_id": None,
            "messages": expected_messages,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("RCA is")
    assert payload["outcome"] == "answered"
    assert payload["grounded"] is True
    assert payload["citations"][0]["source_id"] == "src:pdf/sample"
    assert payload["failure_labels"] == []
    assert payload["active_source_ids"] == ["src:pdf/sample"]
    assert payload["model"] == (
        settings.openai_chat_model
        if settings.llm_backend == "openai_compatible"
        else settings.generation_model
    )


def test_chat_endpoint_rejects_empty_query(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)

    client = TestClient(create_app(settings=settings, graph_store=store))
    response = client.post("/chat", json={"query": "   "})

    assert response.status_code == 400


def test_agent_endpoint_returns_trace_payload(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)

    class FakeAgentLoop:
        def run(self, query: str) -> AgentResult:
            assert query == "inspect my recent experiment runs"
            return AgentResult(
                query=query,
                answer="I inspected recent runs and found 2 failures.",
                trace=AgentTrace(
                    query=query,
                    iterations=2,
                    tool_calls=[],
                    model="stub-agent",
                    total_latency_ms=42.0,
                    stopped_reason="final_answer",
                ),
                error=None,
            )

        def close(self) -> None:
            return None

    client = TestClient(
        create_app(
            settings=settings,
            graph_store=store,
            agent_loop=FakeAgentLoop(),
        )
    )
    response = client.post("/agent", json={"query": "inspect my recent experiment runs"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("I inspected")
    assert payload["trace"]["iterations"] == 2
    assert payload["error"] is None


def test_agent_endpoint_rejects_empty_query(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)

    class FakeAgentLoop:
        def run(self, query: str) -> AgentResult:
            raise AssertionError("should not be called")

        def close(self) -> None:
            return None

    client = TestClient(
        create_app(
            settings=settings,
            graph_store=store,
            agent_loop=FakeAgentLoop(),
        )
    )
    response = client.post("/agent", json={"query": "   "})

    assert response.status_code == 400


def test_ingest_endpoint_accepts_upload_and_returns_result(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)

    class FakeIngestFlow:
        def __init__(self) -> None:
            self.last_path: Path | None = None

        def ingest_path(self, path: str | Path):
            self.last_path = Path(path)
            assert self.last_path.exists()
            assert self.last_path.suffix == ".pdf"
            return IngestResult(
                source_id="src:pdf/uploaded",
                chunk_ids=["chk:pdf/uploaded:0000"],
                node_count=2,
                edge_count=1,
                metadata={"kind": "pdf", "title": "Uploaded"},
            )

    fake_ingest = FakeIngestFlow()
    client = TestClient(
        create_app(
            settings=settings,
            graph_store=store,
            ingest_flow=fake_ingest,
        )
    )

    response = client.post(
        "/sources/ingest",
        files={"file": ("upload.pdf", b"%PDF-1.4\n% test", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_id"] == "src:pdf/uploaded"
    assert payload["chunk_ids"] == ["chk:pdf/uploaded:0000"]
    assert fake_ingest.last_path is not None
    assert fake_ingest.last_path.exists()
    assert fake_ingest.last_path.suffix == ".pdf"


def test_status_endpoint_returns_counts_and_backend_fields(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    vectors = VectorStore(settings.vector_dir, settings.default_collection)

    store.upsert_node(Node(id="src:pdf/p1", kind=NodeKind.paper, title="P1"))
    store.upsert_node(Node(id="chk:pdf/p1:0000", kind=NodeKind.chunk, title="P1 #1"))

    client = TestClient(
        create_app(
            settings=settings,
            graph_store=store,
            vector_store=vectors,
        )
    )
    response = client.get("/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["papers"] == 1
    assert payload["chunks"] == 1
    assert payload["backend"] == "openai_compatible"
    assert payload["model"] == "gpt-4o-mini"
    assert payload["ollama_connected"] is False


def test_status_endpoint_reports_vector_fallback_after_chroma_failure(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    vectors = VectorStore(settings.vector_dir, settings.default_collection)
    client = TestClient(create_app(settings=settings, graph_store=store, vector_store=vectors))

    healthy = client.get("/status").json()
    assert healthy["vector_backend"] == "chroma"
    assert healthy["warnings"] == []

    class BrokenCollection:
        def query(self, **kwargs):
            raise RuntimeError("chroma down")

    vectors._collection = BrokenCollection()
    vectors.query("anything", limit=3)  # real failure path: Chroma error -> JSON fallback

    degraded = client.get("/status").json()
    assert degraded["vector_backend"] == "json"
    assert len(degraded["warnings"]) == 1
    assert "JSON fallback" in degraded["warnings"][0]
    assert "chroma down" in degraded["warnings"][0]
    assert degraded["papers"] == 0  # existing fields unchanged


def test_status_endpoint_clears_the_fallback_warning_once_chroma_works_again(
    tmp_path: Path,
) -> None:
    from tests.unit.test_vector_store import FlakyCollection

    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    vectors = VectorStore(settings.vector_dir, settings.default_collection)
    client = TestClient(create_app(settings=settings, graph_store=store, vector_store=vectors))
    flaky = FlakyCollection()
    vectors._collection = flaky

    flaky.down = True
    vectors.query("anything", limit=3)
    assert client.get("/status").json()["vector_backend"] == "json"

    flaky.down = False
    vectors.query("anything", limit=3)  # no restart
    recovered = client.get("/status").json()
    assert recovered["vector_backend"] == "chroma"
    assert recovered["warnings"] == []


def test_status_endpoint_reports_fallback_of_injected_retrieve_flow_store(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    retrieval_vectors = VectorStore(tmp_path / "retrieval-vectors", settings.default_collection)
    retrieve = RetrieveFlow(settings=settings, graph_store=store, vector_store=retrieval_vectors)
    # No vector_store passed: /status must still describe the store retrieval actually uses.
    client = TestClient(create_app(settings=settings, graph_store=store, retrieve_flow=retrieve))

    class BrokenCollection:
        def query(self, **kwargs):
            raise RuntimeError("chroma down")

    retrieval_vectors._collection = BrokenCollection()
    retrieval_vectors.query("anything", limit=3)

    payload = client.get("/status").json()
    assert payload["vector_backend"] == "json"
    assert any("chroma down" in warning for warning in payload["warnings"])


def test_create_app_shares_retrieve_flow_with_default_agent(tmp_path: Path, monkeypatch) -> None:
    import rca.api.main as api_main

    captured: dict = {}

    class CapturingAgentLoop:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr(api_main, "AgentLoop", CapturingAgentLoop)
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    vectors = VectorStore(settings.vector_dir, settings.default_collection)
    retrieve = RetrieveFlow(settings=settings, graph_store=store, vector_store=vectors)

    create_app(settings=settings, graph_store=store, vector_store=vectors, retrieve_flow=retrieve)

    # Agent knowledge-base search must hit the same store /status reports on.
    assert captured["retrieve_flow"] is retrieve


def test_model_selection_keeps_chat_and_agent_models_separate(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    settings.llm_backend = "openai_compatible"
    settings.openai_chat_model = "chat-model-a"
    settings.agent_model = "agent-model-a"

    class FakeGenerateFlow:
        def __init__(self) -> None:
            self.llm = object()

        def generate_answer(self, query: str, conversation_messages=None):
            return GeneratedAnswer(
                query=query,
                answer="ok",
                citations=[],
                outcome="answered",
                trace=None,
            )

    class FakeAgentLoop:
        def __init__(self) -> None:
            self.llm = object()

        def run(self, query: str) -> AgentResult:
            return AgentResult(
                query=query,
                answer="ok",
                trace=AgentTrace(query=query, iterations=1, tool_calls=[]),
                error=None,
            )

        def close(self) -> None:
            return None

    client = TestClient(
        create_app(
            settings=settings,
            generate_flow=FakeGenerateFlow(),
            agent_loop=FakeAgentLoop(),
        )
    )

    response = client.post("/models/select", json={"model": "chat-model-b"})
    assert response.status_code == 200
    assert response.json()["model"] == "chat-model-b"

    status_response = client.get("/agent/status")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["model"] == "agent-model-a"

    agent_select_response = client.post("/agent/models/select", json={"model": "agent-model-b"})
    assert agent_select_response.status_code == 200
    assert agent_select_response.json()["model"] == "agent-model-b"

    status_response = client.get("/status")
    assert status_response.status_code == 200
    assert status_response.json()["model"] == "chat-model-b"


def test_agent_model_selection_rejects_non_tool_capable_model(tmp_path: Path, monkeypatch) -> None:
    settings = _settings_for(tmp_path)
    settings.llm_backend = "ollama"
    settings.llm_base_url = "http://ollama.local"
    settings.agent_model = "tool-ok:latest"

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_urlopen(request, timeout=0):
        if isinstance(request, str):
            if request.endswith("/api/tags"):
                return _FakeResponse(
                    {"models": [{"name": "tool-ok:latest"}, {"name": "tool-bad:latest"}]}
                )
            if request.endswith("/api/version"):
                return _FakeResponse({"version": "0.0.0"})

        url = request.full_url
        if url.endswith("/api/chat"):
            body = json.loads(request.data.decode("utf-8"))
            if body.get("model") == "tool-bad:latest":
                # Ollama's real refusal names tools; other 400s say nothing about them.
                body = (
                    b'{"error":"registry.ollama.ai/library/tool-bad:latest does not support tools"}'
                )
                raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(body))
            return _FakeResponse({"message": {"content": "ok"}})

        raise AssertionError(f"Unexpected request: {request}")

    monkeypatch.setattr("rca.api.main.urllib.request.urlopen", _fake_urlopen)

    class FakeAgentLoop:
        def run(self, query: str) -> AgentResult:
            return AgentResult(
                query=query,
                answer="ok",
                trace=AgentTrace(query=query, iterations=1, tool_calls=[]),
                error=None,
            )

        def close(self) -> None:
            return None

    client = TestClient(create_app(settings=settings, agent_loop=FakeAgentLoop()))
    response = client.post("/agent/models/select", json={"model": "tool-bad:latest"})

    assert response.status_code == 400
    assert "not tool-calling capable" in response.json()["detail"]


def test_agent_model_diagnostics_exposes_probe_status_and_error_body(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings_for(tmp_path)
    settings.llm_backend = "ollama"
    settings.llm_base_url = "http://ollama.local"
    settings.agent_model = "gemma4:e4b"

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_urlopen(request, timeout=0):
        if isinstance(request, str):
            if request.endswith("/api/tags"):
                return _FakeResponse(
                    {
                        "models": [
                            {"name": "gemma4:e4b", "digest": "sha256:gemma4"},
                            {"name": "phi4-mini:3.8b", "digest": "sha256:phi4mini"},
                        ]
                    }
                )

        url = request.full_url
        if url.endswith("/api/chat"):
            body = json.loads(request.data.decode("utf-8"))
            if body.get("model") == "gemma4:e4b":
                raise urllib.error.HTTPError(
                    url,
                    400,
                    "Bad Request",
                    {},
                    io.BytesIO(b'{"error":"tool schema mismatch"}'),
                )
            return _FakeResponse({"message": {"content": "ok"}})

        raise AssertionError(f"Unexpected request: {request}")

    monkeypatch.setattr("rca.api.main.urllib.request.urlopen", _fake_urlopen)

    class FakeAgentLoop:
        def run(self, query: str) -> AgentResult:
            return AgentResult(
                query=query,
                answer="ok",
                trace=AgentTrace(query=query, iterations=1, tool_calls=[]),
                error=None,
            )

        def close(self) -> None:
            return None

    client = TestClient(create_app(settings=settings, agent_loop=FakeAgentLoop()))
    response = client.get("/agent/models/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    records = payload["records"]

    gemma_record = next(record for record in records if record["name"] == "gemma4:e4b")
    assert gemma_record["probed_tag"] == "gemma4:e4b"
    assert gemma_record["http_status"] == 400
    assert "tool schema mismatch" in gemma_record["error_body"]
    assert gemma_record["present_in_ollama_tags"] is True


def test_agent_model_reprobe_endpoint_refreshes_cache(tmp_path: Path, monkeypatch) -> None:
    settings = _settings_for(tmp_path)
    settings.llm_backend = "ollama"
    settings.llm_base_url = "http://ollama.local"
    settings.agent_model = "gemma4:e4b"

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    probe_counts: dict[str, int] = {}

    def _fake_urlopen(request, timeout=0):
        if isinstance(request, str):
            if request.endswith("/api/tags"):
                return _FakeResponse({"models": [{"name": "gemma4:e4b"}]})

        url = request.full_url
        if url.endswith("/api/chat"):
            body = json.loads(request.data.decode("utf-8"))
            model_name = body.get("model")
            probe_counts[model_name] = probe_counts.get(model_name, 0) + 1
            return _FakeResponse({"message": {"content": "ok"}})

        raise AssertionError(f"Unexpected request: {request}")

    monkeypatch.setattr("rca.api.main.urllib.request.urlopen", _fake_urlopen)

    class FakeAgentLoop:
        def run(self, query: str) -> AgentResult:
            return AgentResult(
                query=query,
                answer="ok",
                trace=AgentTrace(query=query, iterations=1, tool_calls=[]),
                error=None,
            )

        def close(self) -> None:
            return None

    client = TestClient(create_app(settings=settings, agent_loop=FakeAgentLoop()))

    first = client.get("/agent/models")
    assert first.status_code == 200
    assert probe_counts.get("gemma4:e4b", 0) == 1

    second = client.post("/agent/models/reprobe")
    assert second.status_code == 200
    assert probe_counts.get("gemma4:e4b", 0) >= 2


def test_tool_calling_probe_unloads_the_model_right_away(monkeypatch) -> None:
    # The probe runs over every installed model; Ollama would otherwise keep each one
    # in RAM for 5 minutes.
    import io
    import json as _json

    from rca.api import main as api_main
    from rca.config.settings import Settings

    sent: list[dict] = []

    def fake_urlopen(request, timeout=None):
        sent.append(_json.loads(request.data))
        return io.BytesIO(b"{}")

    monkeypatch.setattr(api_main.urllib.request, "urlopen", fake_urlopen)
    from rca.llm.models import probe_tool_calling_capability_detailed

    result = probe_tool_calling_capability_detailed(
        Settings(llm_base_url="http://localhost:11434"), "qwen3:4b"
    )

    assert result["capable"] is True
    assert sent and all(payload["keep_alive"] == 0 for payload in sent)


class _SwitchableReranker:
    """Reranker stub whose model can be broken and repaired, like CrossEncoderReranker."""

    def __init__(self, broken: bool) -> None:
        self.broken = broken
        self.error: str | None = None
        self.loads = 0

    def load(self) -> bool:
        self.loads += 1
        # Hugging Face errors span lines and end in a period; /status shows one clean sentence.
        self.error = (
            "OSError: no cached model, offline.\nSee the offline-mode docs."
            if self.broken
            else None
        )
        return not self.broken

    def rerank(self, query, hits, top_k=None):
        if self.broken:
            self.error = "OSError: no cached model, offline"
            raise OSError("no cached model, offline")
        self.error = None
        return hits


def _app_with_reranker(tmp_path: Path, reranker: _SwitchableReranker):
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    vectors = VectorStore(settings.vector_dir, settings.default_collection)
    retrieve = RetrieveFlow(settings=settings, graph_store=store, vector_store=vectors)
    retrieve._reranker = reranker
    return create_app(
        settings=settings, graph_store=store, vector_store=vectors, retrieve_flow=retrieve
    ), retrieve


def test_status_reports_a_reranker_that_failed_to_load_at_startup(tmp_path: Path) -> None:
    reranker = _SwitchableReranker(broken=True)
    app, _ = _app_with_reranker(tmp_path, reranker)
    with TestClient(app) as client:  # runs startup, before any question
        assert reranker.loads == 1
        warnings = client.get("/status").json()["warnings"]
    assert (
        "Cross-encoder reranker unavailable: OSError: no cached model, offline. Results are not reranked and the no-match gate is off."
        in warnings
    )


def test_status_reranker_warning_clears_after_a_successful_rerank(tmp_path: Path) -> None:
    reranker = _SwitchableReranker(broken=True)
    app, retrieve = _app_with_reranker(tmp_path, reranker)
    client = TestClient(app)
    retrieve.retrieve("anything", limit=3)
    assert any("reranker unavailable" in w for w in client.get("/status").json()["warnings"])

    reranker.broken = False
    retrieve.retrieve("anything", limit=3)
    assert not any("reranker" in w for w in client.get("/status").json()["warnings"])


def test_site_startup_also_loads_the_reranker(tmp_path: Path) -> None:
    from rca.api.main import create_site

    reranker = _SwitchableReranker(broken=False)
    app, _ = _app_with_reranker(tmp_path, reranker)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    with TestClient(create_site(app, frontend_dist=dist)):
        assert reranker.loads == 1


def test_chat_model_is_per_request_and_changes_nothing_shared(tmp_path: Path, monkeypatch) -> None:
    """Two overlapping /chat requests with different models must not answer with each other's."""
    import threading

    from rca.flows.generate_flow import GenerateFlow
    from rca.flows.retrieve_flow import RetrievalBundle, RetrievalHit
    from rca.llm.client import ChatResponse, LLMClient

    class NamedLLM(LLMClient):
        def __init__(self, model: str) -> None:
            self.model = model

        def chat(self, messages):
            return ChatResponse(text=f"Answer from {self.model} [[src:pdf/jampacker]]", raw={})

        def embed(self, texts, dimensions=32):
            raise NotImplementedError

    first_entered, release_first = threading.Event(), threading.Event()

    class BlockingRetrieve:
        graph_store = None
        calls = 0

        def retrieve(self, query, limit=10, trace=None, query_type=None):
            BlockingRetrieve.calls += 1
            if BlockingRetrieve.calls == 1:  # hold the first request inside retrieval
                first_entered.set()
                release_first.wait(timeout=5)
            hit = RetrievalHit(
                node_id="src:pdf/jampacker", score=0.9, title="JamPacker", excerpt="JamPacker packs"
            )
            return RetrievalBundle(query=query, hits=[hit], trace=trace)

    monkeypatch.setattr("rca.api.main.build_llm_for_model", lambda settings, name: NamedLLM(name))
    settings = _settings_for(tmp_path)
    settings.generation_model = "default-model"
    settings.openai_chat_model = "default-model"
    generator = GenerateFlow(
        settings=settings, retrieve_flow=BlockingRetrieve(), llm_client=NamedLLM("default-model")
    )
    client = TestClient(create_app(settings=settings, generate_flow=generator))

    answers: dict[str, dict] = {}
    first = threading.Thread(
        target=lambda: answers.__setitem__(
            "a",
            client.post("/chat", json={"query": "What is JamPacker?", "model": "model-a"}).json(),
        )
    )
    first.start()
    assert first_entered.wait(timeout=5)
    answers["b"] = client.post(
        "/chat", json={"query": "What is JamPacker?", "model": "model-b"}
    ).json()
    release_first.set()
    first.join(timeout=5)

    assert answers["a"]["answer"].startswith("Answer from model-a")
    assert answers["b"]["answer"].startswith("Answer from model-b")
    assert answers["a"]["model"] == "model-a"
    # Nothing shared moved: the default model and the shared client are unchanged.
    assert settings.openai_chat_model == settings.generation_model == "default-model"
    assert generator.llm.model == "default-model"
    plain = client.post("/chat", json={"query": "What is JamPacker?"}).json()
    assert plain["answer"].startswith("Answer from default-model")


class _FakeOllama:
    """Fake Ollama HTTP API: /api/tags lists `tags`; /api/chat probes fail for `no_tools`."""

    def __init__(self, monkeypatch, tags: list[dict], no_tools: set[str] = frozenset()) -> None:
        self.tags = tags
        self.no_tools = set(no_tools)
        self.down = False
        self.probes: list[str] = []
        monkeypatch.setattr("rca.api.main.urllib.request.urlopen", self.urlopen)

    def urlopen(self, request, timeout=0):
        url = request if isinstance(request, str) else request.full_url
        if self.down:
            raise urllib.error.URLError("connection refused")
        if url.endswith("/api/tags"):
            return io.BytesIO(json.dumps({"models": self.tags}).encode())
        if url.endswith("/api/version"):
            return io.BytesIO(b'{"version": "0.0.0"}')
        if url.endswith("/api/chat"):
            model = json.loads(request.data)["model"]
            self.probes.append(model)
            if model in self.no_tools:
                raise urllib.error.HTTPError(
                    url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"does not support tools"}')
                )
            return io.BytesIO(b'{"message": {"content": "ok"}}')
        raise AssertionError(f"Unexpected request: {url}")


class _CountingAgentLoop:
    def __init__(self) -> None:
        self.llm = object()
        self.runs = 0

    def run(self, query: str) -> AgentResult:
        self.runs += 1
        return AgentResult(
            query=query, answer="ok", trace=AgentTrace(query=query, iterations=1), error=None
        )

    def close(self) -> None:
        return None


def _ollama_settings(tmp_path: Path, agent_model: str) -> Settings:
    settings = _settings_for(tmp_path)
    settings.llm_backend = "ollama"
    settings.llm_base_url = "http://ollama.local"
    settings.agent_model = agent_model
    return settings


TWO_MODELS = [
    {"name": "tool-ok:latest", "digest": "sha256:ok"},
    {"name": "tool-bad:latest", "digest": "sha256:bad"},
]


def test_agent_models_lists_installed_models_and_probes_only_the_current_one(
    tmp_path: Path, monkeypatch
) -> None:
    """Opening the picker used to load every installed model into RAM to probe it."""
    ollama = _FakeOllama(monkeypatch, TWO_MODELS, no_tools={"tool-bad:latest"})
    settings = _ollama_settings(tmp_path, "tool-ok:latest")
    client = TestClient(create_app(settings=settings, agent_loop=_CountingAgentLoop()))

    payload = client.get("/agent/models").json()
    assert payload["models"] == ["tool-ok:latest", "tool-bad:latest"]
    assert payload["current"] == "tool-ok:latest"
    assert payload["current_capable"] is True
    client.get("/agent/status")
    assert ollama.probes == ["tool-ok:latest"]


def test_agent_model_is_never_switched_silently(tmp_path: Path, monkeypatch) -> None:
    """Replaces the old auto-reconcile test (PLAN M5): no switch, /agent says what to do."""
    _FakeOllama(monkeypatch, TWO_MODELS, no_tools={"tool-bad:latest"})
    settings = _ollama_settings(tmp_path, "tool-bad:latest")
    agent = _CountingAgentLoop()
    original_llm = agent.llm
    client = TestClient(create_app(settings=settings, agent_loop=agent))

    models = client.get("/agent/models").json()
    assert models["current"] == "tool-bad:latest" and models["current_capable"] is False
    status = client.get("/agent/status").json()
    assert status["model"] == "tool-bad:latest" and status["tool_calling_capable"] is False

    response = client.post("/agent", json={"query": "List the files."})
    assert response.status_code == 409
    assert "tool-bad:latest" in response.json()["detail"]
    assert "cannot call tools (Ollama: does not support tools)" in response.json()["detail"]
    assert agent.runs == 0
    assert settings.agent_model == "tool-bad:latest" and agent.llm is original_llm


def test_tool_probe_is_cached_by_model_digest(tmp_path: Path, monkeypatch) -> None:
    ollama = _FakeOllama(monkeypatch, [{"name": "tool-ok:latest", "digest": "sha256:v1"}])
    settings = _ollama_settings(tmp_path, "tool-ok:latest")
    client = TestClient(create_app(settings=settings, agent_loop=_CountingAgentLoop()))

    client.get("/agent/status")
    client.get("/agent/status")
    assert ollama.probes == ["tool-ok:latest"]
    ollama.tags = [{"name": "tool-ok:latest", "digest": "sha256:v2"}]  # re-pulled
    client.get("/agent/status")
    assert ollama.probes == ["tool-ok:latest", "tool-ok:latest"]


def test_an_unreachable_ollama_is_not_taken_for_a_model_without_tools(
    tmp_path: Path, monkeypatch
) -> None:
    ollama = _FakeOllama(monkeypatch, TWO_MODELS)
    settings = _ollama_settings(tmp_path, "tool-ok:latest")
    agent = _CountingAgentLoop()
    client = TestClient(create_app(settings=settings, agent_loop=agent))

    ollama.down = True
    assert client.get("/agent/status").json()["tool_calling_capable"] is None
    assert client.post("/agent", json={"query": "List the files."}).status_code == 200
    ollama.down = False
    assert client.get("/agent/status").json()["tool_calling_capable"] is True


def test_only_a_refusal_about_tools_is_cached_as_no_tools(tmp_path: Path, monkeypatch) -> None:
    """A 5xx (e.g. not enough memory to load it) says nothing about tools; retry next time."""
    from rca.llm.models import tool_capability

    ollama = _FakeOllama(monkeypatch, [{"name": "qwen3.5:4b", "digest": "sha256:d1"}])
    settings = _ollama_settings(tmp_path, "qwen3.5:4b")
    cache: dict = {}
    original = ollama.urlopen

    def out_of_memory(request, timeout=0):
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/api/chat"):
            raise urllib.error.HTTPError(
                url,
                500,
                "Internal",
                {},
                io.BytesIO(b'{"error":"model requires more system memory"}'),
            )
        return original(request, timeout)

    monkeypatch.setattr("rca.api.main.urllib.request.urlopen", out_of_memory)
    assert tool_capability(settings, cache, "qwen3.5:4b") is None
    assert cache == {}
    monkeypatch.setattr("rca.api.main.urllib.request.urlopen", original)
    assert tool_capability(settings, cache, "qwen3.5:4b") is True


def test_the_refusal_reason_is_the_current_digests(tmp_path: Path, monkeypatch) -> None:
    from rca.llm.models import tool_capability, tool_refusal_reason

    ollama = _FakeOllama(monkeypatch, [], no_tools={"m:latest"})
    settings = _ollama_settings(tmp_path, "m:latest")
    cache: dict = {"m:latest": {"capable": False, "error_body": '{"error":"model not found"}'}}
    ollama.tags = [{"name": "m:latest", "digest": "sha256:d2"}]  # pulled since
    assert tool_capability(settings, cache, "m:latest") is False
    assert tool_refusal_reason(settings, cache, "m:latest") == "does not support tools"
