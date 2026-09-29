from __future__ import annotations

import logging
from pathlib import Path

from rca.llm.embeddings import ConfigurableEmbeddingFunction
from rca.store.vector_store import VectorStore


def test_vector_store_logs_warning_when_chroma_query_falls_back(tmp_path, caplog) -> None:
    class BrokenCollection:
        def query(self, **kwargs):
            raise RuntimeError("simulated chroma outage")

    store = object.__new__(VectorStore)
    store.persist_dir = tmp_path
    store.collection_name = "test"
    store._fallback_path = Path(tmp_path / "test.json")
    store._documents = {
        "doc-1": {
            "document": "research continuity memory",
            "metadata": {"source_id": "src:note/demo"},
        }
    }
    store._collection = BrokenCollection()
    store._chroma_error = None

    with caplog.at_level(logging.WARNING):
        results = store.query("research continuity", limit=5)

    assert len(results) == 1
    assert results[0].id == "doc-1"
    assert store.backend == "json"
    assert "Disabling Chroma backend during query" in caplog.text
    assert "RuntimeError: simulated chroma outage" in caplog.text


def test_configurable_embedding_function_uses_client_embed_dimensions() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], int]] = []

        def embed(self, texts: list[str], dimensions: int = 32) -> list[list[float]]:
            self.calls.append((texts, dimensions))
            return [[0.1, 0.2], [0.3, 0.4]]

    client = FakeClient()
    embedding_fn = ConfigurableEmbeddingFunction(client=client, dimensions=768)

    vectors = embedding_fn(["alpha", "beta"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.calls == [(["alpha", "beta"], 768)]


def test_vector_store_delete_texts_removes_json_fallback_records(tmp_path) -> None:
    store = object.__new__(VectorStore)
    store.persist_dir = tmp_path
    store.collection_name = "test"
    store._fallback_path = Path(tmp_path / "test.json")
    store._documents = {
        "doc-1": {"document": "alpha", "metadata": {"source_id": "src:note/demo"}},
        "doc-2": {"document": "beta", "metadata": {"source_id": "src:note/demo"}},
    }
    store._collection = None
    store._chroma_error = None

    store.delete_texts(["doc-2"])

    assert "doc-1" in store._documents
    assert "doc-2" not in store._documents


def test_vector_store_delete_texts_fail_closed_on_chroma_error(tmp_path) -> None:
    class BrokenDeleteCollection:
        def delete(self, **kwargs):
            raise RuntimeError("simulated delete failure")

    store = object.__new__(VectorStore)
    store.persist_dir = tmp_path
    store.collection_name = "test"
    store._fallback_path = Path(tmp_path / "test.json")
    store._documents = {
        "doc-1": {"document": "alpha", "metadata": {"source_id": "src:note/demo"}},
    }
    store._collection = BrokenDeleteCollection()
    store._chroma_error = None

    try:
        store.delete_texts(["doc-1"])
        assert False, "delete_texts should raise when Chroma delete fails"
    except RuntimeError as exc:
        assert "simulated delete failure" in str(exc)

    assert "doc-1" in store._documents
