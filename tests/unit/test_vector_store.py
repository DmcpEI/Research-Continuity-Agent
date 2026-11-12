from __future__ import annotations

import json
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
    assert "Chroma failed during query" in caplog.text
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

    # Still fails closed, but the pending copy is gone, so no resync can bring the chunk back.
    assert "doc-1" not in store._documents


def test_backend_status_summarizes_distinct_stores_and_dedupes_warnings() -> None:
    from types import SimpleNamespace

    from rca.store.vector_store import backend_status

    healthy = SimpleNamespace(backend="chroma", backend_warning=None)
    fallen = SimpleNamespace(
        backend="json", backend_warning="VectorStore using JSON fallback backend."
    )
    fallen_twin = SimpleNamespace(backend="json", backend_warning=fallen.backend_warning)

    assert backend_status([healthy, healthy, None]) == ("chroma", [])
    # same instance twice and two stores with the same message -> one warning
    assert backend_status([healthy, fallen, fallen, fallen_twin]) == (
        "json",
        ["VectorStore using JSON fallback backend."],
    )
    assert backend_status([]) == ("unknown", [])


class FlakyCollection:
    """Fake Chroma collection: fails while `down` is true, records what reaches it."""

    def __init__(self) -> None:
        self.down = False
        self.records: dict[str, tuple[str, dict]] = {}

    def _check(self) -> None:
        if self.down:
            raise ConnectionError("embedding server unreachable")

    def query(self, query_texts, n_results, include):
        self._check()
        ids = list(self.records)[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.records[i][0] for i in ids]],
            "metadatas": [[self.records[i][1] for i in ids]],
            "distances": [[0.1 for _ in ids]],
        }

    def upsert(self, ids, documents, metadatas, **kwargs):
        self._check()
        for record_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self.records[record_id] = (document, metadata or {})

    def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)

    def count(self) -> int:
        return len(self.records)


def _flaky_store(tmp_path, documents=None) -> tuple[VectorStore, FlakyCollection]:
    store = object.__new__(VectorStore)
    store.persist_dir = tmp_path
    store.collection_name = "test"
    store._fallback_path = Path(tmp_path / "test.json")
    store._documents = dict(documents or {})
    collection = FlakyCollection()
    store._collection = collection
    return store, collection


def test_vector_store_returns_to_chroma_after_a_transient_failure(tmp_path) -> None:
    store, collection = _flaky_store(tmp_path)
    collection.records["chk:a:0"] = ("alpha paper", {"source_id": "src:a"})

    collection.down = True
    store.query("alpha", limit=3)
    assert store.backend == "json"
    assert "embedding server unreachable" in (store.backend_warning or "")

    collection.down = False
    results = store.query("alpha", limit=3)
    assert [result.id for result in results] == ["chk:a:0"]
    assert store.backend == "chroma"
    assert store.backend_warning is None
    assert store.fallback_events == 1


def test_writes_made_while_degraded_reach_chroma_on_recovery(tmp_path) -> None:
    store, collection = _flaky_store(tmp_path)
    collection.down = True
    store.upsert_texts(["chk:new:0"], ["new chunk"], [{"source_id": "src:new"}])
    assert "chk:new:0" not in collection.records
    assert store._fallback_path.exists()

    collection.down = False
    store.query("anything", limit=3)
    assert collection.records["chk:new:0"] == ("new chunk", {"source_id": "src:new"})
    assert store._documents == {}
    assert json.loads(store._fallback_path.read_text(encoding="utf-8")) == {}


def test_degraded_writes_left_from_a_previous_process_are_synced(tmp_path) -> None:
    pending = {
        "chk:old:0": {"document": "written while degraded", "metadata": {"source_id": "src:old"}}
    }
    store, collection = _flaky_store(tmp_path, pending)
    store.query("anything", limit=3)
    assert "chk:old:0" in collection.records
    assert store._documents == {}


def test_a_failed_resync_keeps_the_pending_writes_and_stays_degraded(tmp_path) -> None:
    store, collection = _flaky_store(tmp_path)
    collection.down = True
    store.upsert_texts(["chk:new:0"], ["new chunk"], [{"source_id": "src:new"}])

    original_upsert = collection.upsert

    def failing_upsert(**kwargs):
        raise RuntimeError("disk full")

    collection.down = False
    collection.upsert = failing_upsert
    try:
        store.query("anything", limit=3)
    finally:
        collection.upsert = original_upsert
    assert "chk:new:0" in store._documents
    assert store.backend == "json"
    assert "disk full" in (store.backend_warning or "")


def test_a_delete_while_degraded_is_not_resurrected_by_the_resync(tmp_path) -> None:
    store, collection = _flaky_store(tmp_path)
    collection.records["chk:old:0"] = ("old chunk", {"source_id": "src:old"})
    collection.down = True
    store.upsert_texts(["chk:new:0"], ["new chunk"], [{"source_id": "src:new"}])
    store.delete_texts(["chk:old:0", "chk:new:0"])  # delete needs no embeddings

    collection.down = False
    store.query("anything", limit=3)
    assert collection.records == {}


def test_a_delete_racing_a_resync_does_not_resurrect_the_chunk(tmp_path) -> None:
    import threading

    store, collection = _flaky_store(tmp_path)
    collection.down = True
    store.upsert_texts(["chk:x:0"], ["doomed chunk"], [{"source_id": "src:x"}])
    collection.down = False

    deleted_in_chroma = threading.Event()
    original_delete, original_upsert = collection.delete, collection.upsert

    def tracking_delete(ids):
        original_delete(ids)
        deleted_in_chroma.set()

    collection.delete = tracking_delete
    deleter: list[threading.Thread] = []

    def upsert_while_a_delete_starts(**kwargs):
        # The resync holds the store; a DELETE /sources arrives now on another thread.
        if not deleter:
            deleter.append(threading.Thread(target=store.delete_texts, args=(["chk:x:0"],)))
            deleter[0].start()
            deleted_in_chroma.wait(timeout=0.5)  # old code deletes in Chroma here, too early
        original_upsert(**kwargs)

    collection.upsert = upsert_while_a_delete_starts
    store.query("anything", limit=3)
    deleter[0].join(timeout=5)
    assert "chk:x:0" not in collection.records


def test_a_failed_chroma_delete_still_drops_the_pending_copy(tmp_path) -> None:
    store, collection = _flaky_store(tmp_path)
    collection.down = True
    store.upsert_texts(["chk:x:0"], ["doomed chunk"], [{"source_id": "src:x"}])

    def broken_delete(ids):
        raise RuntimeError("delete failed")

    collection.delete = broken_delete
    try:
        store.delete_texts(["chk:x:0"])
    except RuntimeError:
        pass
    collection.down = False
    store.query("anything", limit=3)
    assert "chk:x:0" not in collection.records


def test_a_chunk_chroma_rejects_is_set_aside_and_the_rest_resync(tmp_path, caplog) -> None:
    pending = {
        "chk:bad:0": {"document": "bad", "metadata": {"tags": ["a"]}},
        "chk:good:0": {"document": "good", "metadata": {"source_id": "src:good"}},
    }
    store, collection = _flaky_store(tmp_path, pending)
    original_upsert = collection.upsert

    def strict_upsert(ids, documents, metadatas, **kwargs):
        if "chk:bad:0" in ids:
            raise ValueError("Expected metadata value to be a str, int, float or bool")
        original_upsert(ids=ids, documents=documents, metadatas=metadatas)

    collection.upsert = strict_upsert
    with caplog.at_level(logging.WARNING):
        store.query("anything", limit=3)
    assert "chk:good:0" in collection.records
    assert store.backend == "chroma" and store.backend_warning is None
    rejected = json.loads((tmp_path / "test.rejected.json").read_text(encoding="utf-8"))
    assert list(rejected) == ["chk:bad:0"]
    assert "set aside" in caplog.text
    assert "serving this call from the JSON store" not in caplog.text


def test_a_large_backlog_resyncs_a_batch_per_call_and_says_so(tmp_path) -> None:
    pending = {
        f"chk:p:{i}": {"document": f"chunk {i}", "metadata": {"source_id": "src:p"}}
        for i in range(130)
    }
    store, collection = _flaky_store(tmp_path, pending)
    store.query("anything", limit=3)
    assert len(collection.records) == 64
    assert store.backend == "chroma"
    assert "66 chunks" in (store.backend_warning or "")
    store.query("anything", limit=3)
    store.query("anything", limit=3)
    assert len(collection.records) == 130
    assert store.backend_warning is None
