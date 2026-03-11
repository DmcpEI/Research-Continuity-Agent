"""Chroma-compatible vector storage with a local JSON fallback."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:
    chromadb = None


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class VectorQueryResult:
    id: str
    score: float
    document: str
    metadata: dict[str, Any]


class VectorStore:
    """Persist text chunks in Chroma when available, otherwise use a JSON fallback."""

    def __init__(self, persist_dir: str | Path, collection_name: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._fallback_path = self.persist_dir / f"{self.collection_name}.json"
        self._documents = self._load_fallback_documents()
        self._collection = None

        if chromadb is not None:
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
            from rca.config.settings import get_settings
            settings = get_settings()

            embedding_fn = OllamaEmbeddingFunction(
                url=f"{settings.embedding_base_url}/api/embeddings",
                model_name=settings.embedding_model,
            )

            client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

    @property
    def backend(self) -> str:
        return "chroma" if self._collection is not None else "json"

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
            self._collection.upsert(**payload)
            return

        for record_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self._documents[record_id] = {"document": document, "metadata": metadata}
        self._fallback_path.write_text(json.dumps(self._documents, indent=2, sort_keys=True), encoding="utf-8")

    def query(self, query_text: str, limit: int = 5) -> list[VectorQueryResult]:
        if self._collection is not None:
            result = self._collection.query(
                query_texts=[query_text],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            ids = result.get("ids", [[]])[0]
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            return [
                VectorQueryResult(
                    id=record_id,
                    score=1.0 - float(distance),
                    document=document,
                    metadata=metadata or {},
                )
                for record_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=True)
            ]

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

    def _load_fallback_documents(self) -> dict[str, dict[str, Any]]:
        if not self._fallback_path.exists():
            return {}
        return json.loads(self._fallback_path.read_text(encoding="utf-8"))

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
