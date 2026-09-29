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
        event_log_path=runtime_dir / "events.jsonl",
        telemetry_log_path=runtime_dir / "telemetry.jsonl",
        experiment_db_path=runtime_dir / "experiments.sqlite3",
        tool_policy_path=Path("rca/config/tool_policies.yaml"),
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
                grounded=True,
                abstained=False,
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


def test_agent_models_filters_out_non_tool_calling_models(tmp_path: Path, monkeypatch) -> None:
    settings = _settings_for(tmp_path)
    settings.llm_backend = "ollama"
    settings.llm_base_url = "http://ollama.local"
    settings.agent_model = "tool-ok:latest"
    settings.generation_model = "chat-default:latest"

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
                raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b"{}"))
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
    response = client.get("/agent/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["tool-ok:latest"]
    assert payload["current"] == "tool-ok:latest"


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
                grounded=True,
                abstained=False,
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


def test_agent_models_auto_reconciles_invalid_current_model(tmp_path: Path, monkeypatch) -> None:
    settings = _settings_for(tmp_path)
    settings.llm_backend = "ollama"
    settings.llm_base_url = "http://ollama.local"
    settings.agent_model = "missing-model:latest"

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
            if body.get("model") in {"tool-bad:latest", "missing-model:latest"}:
                raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b"{}"))
            return _FakeResponse({"message": {"content": "ok"}})

        raise AssertionError(f"Unexpected request: {request}")

    monkeypatch.setattr("rca.api.main.urllib.request.urlopen", _fake_urlopen)

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

    client = TestClient(create_app(settings=settings, agent_loop=FakeAgentLoop()))

    response = client.get("/agent/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["tool-ok:latest"]
    assert payload["current"] == "tool-ok:latest"

    status_response = client.get("/agent/status")
    assert status_response.status_code == 200
    assert status_response.json()["model"] == "tool-ok:latest"


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
                raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b"{}"))
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
