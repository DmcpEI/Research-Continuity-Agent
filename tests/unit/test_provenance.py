from __future__ import annotations

import io
import json
from types import SimpleNamespace

from rca.contracts.nodes import Node, NodeKind
from rca.store.graph_store import GraphStore
from rca.telemetry import provenance


def _store(tmp_path, chunk_text: str, order=("src", "chk")) -> GraphStore:
    store = GraphStore(tmp_path / "graph.sqlite3")
    nodes = {
        "src": Node(id="src:pdf/a", kind=NodeKind.paper, title="A", text="abstract"),
        "chk": Node(id="chk:pdf/a:0000", kind=NodeKind.chunk, title="A #1", text=chunk_text),
    }
    for key in order:
        store.upsert_node(nodes[key])
    return store


def test_corpus_fingerprint_tracks_content_not_insertion_order(tmp_path) -> None:
    base = provenance.corpus_fingerprint(_store(tmp_path / "a", "alpha"))
    reordered = provenance.corpus_fingerprint(_store(tmp_path / "b", "alpha", ("chk", "src")))
    edited = provenance.corpus_fingerprint(_store(tmp_path / "c", "alpha!"))

    assert base == reordered
    assert base["sources"] == 1 and base["chunks"] == 1
    assert edited["corpus_version"] != base["corpus_version"]


def test_ollama_models_maps_digests_and_survives_an_unreachable_server(monkeypatch) -> None:
    responses = {
        "/api/version": {"version": "0.35.1"},
        "/api/tags": {
            "models": [
                {"name": "gemma3:12b", "digest": "f4031aab"},
                {"name": "nomic-embed-text:latest", "digest": "0a109f42"},
                {"name": 7},  # malformed entry from outside is ignored
            ]
        },
    }

    def fake_urlopen(url, timeout):
        path = url.split("11434", 1)[1]
        return io.BytesIO(json.dumps(responses[path]).encode())

    monkeypatch.setattr(provenance.urllib.request, "urlopen", fake_urlopen)
    result = provenance.ollama_models(
        "http://localhost:11434", ["gemma3:12b", "nomic-embed-text", "qwen2.5:14b"]
    )
    assert result == {
        "ollama_version": "0.35.1",
        "models": {"gemma3:12b": "f4031aab", "nomic-embed-text": "0a109f42", "qwen2.5:14b": None},
    }

    def refused(url, timeout):
        raise ConnectionRefusedError(61, "Connection refused")

    monkeypatch.setattr(provenance.urllib.request, "urlopen", refused)
    down = provenance.ollama_models("http://localhost:11434", ["gemma3:12b"])
    assert down["models"] == {"gemma3:12b": None}
    assert "ConnectionRefusedError" in down["error"]


def test_finish_provenance_flags_a_mid_run_vector_fallback() -> None:
    store = SimpleNamespace(backend="chroma", backend_warning=None, _collection=None, _documents={})
    started = {"index_start": provenance.index_snapshot(store)}

    assert provenance.finish_provenance(started, store)["index_degraded_during_run"] is False

    store.backend = "json"
    store.backend_warning = "VectorStore using JSON fallback after Chroma failure: x"
    finished = provenance.finish_provenance(started, store)
    assert finished["index_degraded_during_run"] is True
    assert finished["index_end"]["backend"] == "json"


def test_git_state_reports_commit_of_this_repo() -> None:
    state = provenance.git_state()
    assert state["commit"] and len(state["commit"]) == 40
    assert isinstance(state["dirty"], bool)


def test_collect_provenance_tolerates_missing_stores() -> None:
    settings = SimpleNamespace(
        llm_backend="openai_compatible",
        generation_model="g",
        agent_model="a",
        embedding_model="e",
        openai_chat_model="gpt",
        openai_embed_model="emb",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    result = provenance.finish_provenance(provenance.collect_provenance(settings))
    assert result["corpus"] is None and result["index_start"] is None
    assert result["index_end"] is None and result["index_degraded_during_run"] is None
    assert result["models"]["models"] == {"gpt": None, "emb": None}
    assert result["settings"]["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_corpus_fingerprint_covers_titles_and_edges_and_never_raises(tmp_path) -> None:
    from rca.contracts.nodes import Edge, EdgeKind

    base = provenance.corpus_fingerprint(_store(tmp_path / "a", "alpha"))

    retitled = _store(tmp_path / "b", "alpha")
    retitled.upsert_node(
        Node(id="src:pdf/a", kind=NodeKind.paper, title="Renamed", text="abstract")
    )
    assert provenance.corpus_fingerprint(retitled)["corpus_version"] != base["corpus_version"]

    linked = _store(tmp_path / "c", "alpha")
    linked.upsert_edge(Edge(source="src:pdf/a", target="chk:pdf/a:0000", kind=EdgeKind.contains))
    assert provenance.corpus_fingerprint(linked)["corpus_version"] != base["corpus_version"]

    class LockedStore:
        def connect(self):
            raise RuntimeError("database is locked")

    broken = provenance.corpus_fingerprint(LockedStore())
    assert broken == {"error": "RuntimeError: database is locked"}


def test_index_degradation_means_a_change_during_the_run() -> None:
    json_from_start = SimpleNamespace(
        backend="json",
        backend_warning="VectorStore using JSON fallback backend.",
        _collection=None,
        _documents={},
    )
    started = {"index_start": provenance.index_snapshot(json_from_start)}
    finished = provenance.finish_provenance(started, json_from_start)
    assert finished["index_degraded_during_run"] is False
    assert finished["index_start"]["backend"] == "json"  # still visible for the reader

    bare = object()  # a test double without backend attributes must not crash the run
    assert provenance.index_snapshot(bare)["backend"] is None


def test_ollama_models_keeps_installed_models_that_lack_a_digest(monkeypatch) -> None:
    payloads = {"/api/version": {"version": "x"}, "/api/tags": {"models": [{"name": "gemma3:12b"}]}}
    monkeypatch.setattr(
        provenance.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(json.dumps(payloads[url.split("11434", 1)[1]]).encode()),
    )
    result = provenance.ollama_models("http://localhost:11434", ["gemma3:12b", "missing:1b"])
    assert result["models"] == {"gemma3:12b": "", "missing:1b": None}


def test_git_state_hashes_uncommitted_changes_when_dirty() -> None:
    state = provenance.git_state()
    if state["dirty"]:
        assert len(state["diff_sha256"]) == 16
    else:
        assert state["diff_sha256"] is None
