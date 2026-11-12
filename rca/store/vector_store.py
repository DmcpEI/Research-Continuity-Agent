"""Chroma-compatible vector storage with a local JSON fallback."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:
    chromadb = None


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
# Chunks copied into Chroma per successful call after an outage, so a large backlog never holds
# the store (and a user's request) for one long embedding run.
RESYNC_BATCH = 64
LOGGER = logging.getLogger(__name__)


def backend_status(stores: Iterable[Any]) -> tuple[str, list[str]]:
    """Summarize distinct vector stores as (backend, unique warnings).

    Backend is "json" if any store is on its fallback, "chroma" if all are healthy,
    "unknown" if there are none. A store can change state between the two reads while a
    call is running; the next /status poll shows the settled state.
    """
    unique = list({id(store): store for store in stores if store is not None}.values())
    warnings = [store.backend_warning for store in unique]
    backends = [store.backend for store in unique]
    if not backends:
        return "unknown", []
    backend = "json" if "json" in backends else "chroma"
    return backend, list(dict.fromkeys(warning for warning in warnings if warning))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class VectorQueryResult:
    id: str
    score: float
    document: str
    metadata: dict[str, Any]


class VectorStore:
    """Persist text chunks in Chroma when available, otherwise use a JSON fallback.

    Chroma is tried on every call. A failed call is served from the JSON store and the store
    reports itself degraded until the next successful Chroma call. With Chroma installed, the
    JSON file holds only chunks written while degraded; the next success copies them into Chroma
    and empties it. Without Chroma, the JSON file is the whole index.
    """

    # Class defaults keep stores built with object.__new__ (tests) valid.
    _chroma_error: str | None = None  # last failure, while degraded
    _degraded_since: str | None = None
    fallback_events: int = (
        0  # calls served from JSON after a Chroma failure; provenance compares it
    )
    _lock = threading.RLock()

    def __init__(self, persist_dir: str | Path, collection_name: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._fallback_path = self.persist_dir / f"{self.collection_name}.json"
        self._documents = self._load_fallback_documents()
        self._collection = None
        self._lock = threading.RLock()

        if chromadb is not None:
            from rca.config.settings import get_settings
            from rca.llm.embeddings import ConfigurableEmbeddingFunction
            from rca.llm.factory import get_llm_client

            settings = get_settings()
            embedding_fn = ConfigurableEmbeddingFunction(
                client=get_llm_client(settings),
                dimensions=settings.embedding_dimensions,
            )

            client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

    @property
    def backend(self) -> str:
        return "chroma" if self._collection is not None and self._chroma_error is None else "json"

    @property
    def backend_warning(self) -> str | None:
        error = self._chroma_error
        if self._collection is None:
            return "VectorStore using JSON fallback backend."
        if error:
            return (
                f"VectorStore using JSON fallback after Chroma failure since "
                f"{self._degraded_since}: {error}"
            )
        waiting = len(self._documents)
        if waiting:
            return (
                f"VectorStore copying {waiting} chunks written during a Chroma outage into the "
                "index; until then vector search does not see them."
            )
        return None

    def upsert_texts(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        metadatas = metadatas or [{} for _ in ids]

        if self._collection is not None:
            payload: dict[str, Any] = {"ids": ids, "documents": documents, "metadatas": metadatas}
            if embeddings is not None:
                payload["embeddings"] = embeddings
            try:
                self._collection.upsert(**payload)
            except Exception as exc:
                self._fall_back(exc, operation="upsert")
            else:
                self._chroma_succeeded()
                return

        with self._lock:
            self._upsert_fallback(ids, documents, metadatas)

    def query(self, query_text: str, limit: int = 5) -> list[VectorQueryResult]:
        if self._collection is not None:
            try:
                result = self._collection.query(
                    query_texts=[query_text],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                )
                ids = result.get("ids", [[]])[0]
                documents = result.get("documents", [[]])[0]
                metadatas = result.get("metadatas", [[]])[0]
                distances = result.get("distances", [[]])[0]
                results = [
                    VectorQueryResult(
                        id=record_id,
                        score=1.0 - float(distance),
                        document=document,
                        metadata=metadata or {},
                    )
                    for record_id, document, metadata, distance in zip(
                        ids, documents, metadatas, distances, strict=True
                    )
                ]
            except Exception as exc:
                self._fall_back(exc, operation="query")
            else:
                self._chroma_succeeded()
                return results

        return self._query_fallback(query_text, limit)

    def delete_texts(self, ids: list[str]) -> None:
        if not ids:
            return

        # Under the lock, so a resync cannot copy a chunk into Chroma after it was deleted there.
        with self._lock:
            # Pending copies go first: if the Chroma delete fails, no resync may bring them back.
            self._delete_fallback(ids)
            if self._collection is not None:
                # Fail closed on Chroma delete failures to avoid serving stale vectors. Delete
                # needs no embeddings, so this also runs while queries are degraded.
                self._collection.delete(ids=ids)

    def _load_fallback_documents(self) -> dict[str, dict[str, Any]]:
        if not self._fallback_path.exists():
            return {}
        return json.loads(self._fallback_path.read_text(encoding="utf-8"))

    def _upsert_fallback(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for record_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self._documents[record_id] = {"document": document, "metadata": metadata}
        self._fallback_path.write_text(
            json.dumps(self._documents, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _delete_fallback(self, ids: list[str]) -> None:
        removed = False
        for record_id in ids:
            if record_id in self._documents:
                removed = True
                del self._documents[record_id]
        # ponytail: the file is rewritten from this process's copy, so two processes writing to
        # one library while degraded can overwrite each other; write only when something changed.
        if removed:
            self._fallback_path.write_text(
                json.dumps(self._documents, indent=2, sort_keys=True), encoding="utf-8"
            )

    def _query_fallback(self, query_text: str, limit: int) -> list[VectorQueryResult]:
        query_counter = Counter(self._tokenize(query_text))
        results: list[VectorQueryResult] = []
        for record_id, payload in self._documents.items():
            document_counter = Counter(self._tokenize(payload["document"]))
            score = self._cosine_overlap(query_counter, document_counter)
            if score > 0:
                results.append(
                    VectorQueryResult(
                        id=record_id,
                        score=score,
                        document=payload["document"],
                        metadata=payload["metadata"],
                    )
                )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _fall_back(self, exc: Exception, operation: str) -> None:
        self._chroma_error = f"{type(exc).__name__}: {exc}"
        self._degraded_since = self._degraded_since or _utc_now()
        self.fallback_events += 1
        LOGGER.warning(
            "Chroma failed during %s; serving this call from the JSON store (%s)",
            operation,
            self._chroma_error,
        )

    def _chroma_succeeded(self) -> None:
        """Chroma just worked: copy the next batch of pending chunks into it, then report healthy."""
        if self._chroma_error is None and not self._documents:
            return
        with self._lock:
            batch = dict(list(self._documents.items())[:RESYNC_BATCH])
            if batch:
                try:
                    synced, rejected = self._resync(batch)
                except Exception as exc:
                    self._chroma_error = f"{type(exc).__name__}: {exc}"
                    self._degraded_since = self._degraded_since or _utc_now()
                    self.fallback_events += 1
                    LOGGER.warning(
                        "Copying pending chunks into Chroma failed; will retry (%s)", exc
                    )
                    return
                if rejected:
                    self._set_aside(rejected)
                self._delete_fallback([*synced, *rejected])
            if self._chroma_error is not None:
                LOGGER.warning("Chroma is back after failing since %s", self._degraded_since)
            self._chroma_error = None
            self._degraded_since = None

    def _resync(self, batch: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
        """Upsert a batch; on a validation error, find and return the chunks Chroma rejects.

        Other errors (the embedding server, timeouts) propagate: the batch stays pending.
        """

        def upsert(items: dict[str, dict[str, Any]]) -> None:
            self._collection.upsert(  # type: ignore[union-attr]
                ids=list(items),
                documents=[payload["document"] for payload in items.values()],
                metadatas=[payload["metadata"] or None for payload in items.values()],
            )

        try:
            upsert(batch)
            return list(batch), {}
        except (ValueError, TypeError):
            pass
        synced: list[str] = []
        rejected: dict[str, Any] = {}
        for record_id, payload in batch.items():
            try:
                upsert({record_id: payload})
            except (ValueError, TypeError) as exc:
                rejected[record_id] = payload | {"error": f"{type(exc).__name__}: {exc}"}
            else:
                synced.append(record_id)
        return synced, rejected

    def _set_aside(self, rejected: dict[str, Any]) -> None:
        """Keep chunks Chroma refuses in a side file instead of retrying them forever."""
        path = self.persist_dir / f"{self.collection_name}.rejected.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        path.write_text(json.dumps(existing | rejected, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.warning(
            "Chroma rejected %d pending chunks; set aside in %s: %s",
            len(rejected),
            path,
            sorted(rejected),
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _cosine_overlap(left: Counter[str], right: Counter[str]) -> float:
        numerator = sum(left[token] * right[token] for token in left.keys() & right.keys())
        if numerator == 0:
            return 0.0
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)
