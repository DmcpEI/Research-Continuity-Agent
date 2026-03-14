from __future__ import annotations

import logging
from pathlib import Path

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
