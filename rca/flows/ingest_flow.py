"""Business logic for ingesting artifacts into local stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rca.config.settings import Settings, get_settings
from rca.contracts.ids import make_chunk_id, make_source_id
from rca.contracts.nodes import Edge, EdgeKind, Node, NodeKind
from rca.extractors.experiment_extractor import ExperimentExtractor
from rca.extractors.git_extractor import GitExtractor
from rca.extractors.note_extractor import NoteExtractor
from rca.extractors.pdf_extractor import PDFExtractor
from rca.store.event_log import EventLog, IngestEvent
from rca.store.graph_store import GraphStore
from rca.store.vector_store import VectorStore


class IngestResult(BaseModel):
    source_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestFlow:
    """Ingest deterministic source material into graph and vector stores."""

    def __init__(
        self,
        settings: Settings | None = None,
        graph_store: GraphStore | None = None,
        vector_store: VectorStore | None = None,
        event_log: EventLog | None = None,
        note_extractor: NoteExtractor | None = None,
        pdf_extractor: PDFExtractor | None = None,
        git_extractor: GitExtractor | None = None,
        experiment_extractor: ExperimentExtractor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_runtime_directories()
        self.graph_store = graph_store or GraphStore(self.settings.graph_db_path)
        self.vector_store = vector_store or VectorStore(self.settings.vector_dir, self.settings.default_collection)
        self.event_log = event_log or EventLog(self.settings.event_log_path)
        self.note_extractor = note_extractor or NoteExtractor()
        self.pdf_extractor = pdf_extractor or PDFExtractor()
        self.git_extractor = git_extractor or GitExtractor()
        self.experiment_extractor = experiment_extractor or ExperimentExtractor()

    def ingest_path(self, path: str | Path) -> IngestResult:
        source_path = Path(path)
        payload = self._extract(source_path)
        source_id = make_source_id(payload["metadata"]["kind"], source_path.stem if source_path.is_file() else source_path.name)
        source_kind = self._source_kind(payload["metadata"]["kind"])

        source_node = Node(
            id=source_id,
            kind=source_kind,
            title=payload["title"],
            text=payload["content"] or None,
            metadata=payload["metadata"],
        )
        self.graph_store.upsert_node(source_node)

        chunk_ids: list[str] = []
        vector_ids: list[str] = []
        vector_documents: list[str] = []
        vector_metadatas: list[dict[str, Any]] = []

        for index, chunk_text in enumerate(self._chunk_content(payload["content"])):
            chunk_id = make_chunk_id(source_id, index)
            chunk_ids.append(chunk_id)

            chunk_node = Node(
                id=chunk_id,
                kind=NodeKind.chunk,
                title=f"{payload['title']} #{index + 1}",
                text=chunk_text,
                metadata={"source_id": source_id, "chunk_index": index, "kind": payload["metadata"]["kind"]},
            )
            self.graph_store.upsert_node(chunk_node)
            self.graph_store.upsert_edge(
                Edge(
                    source=source_id,
                    target=chunk_id,
                    kind=EdgeKind.contains,
                    metadata={"sequence": index},
                )
            )

            vector_ids.append(chunk_id)
            vector_documents.append(chunk_text)
            vector_metadatas.append({"source_id": source_id, "title": payload["title"]})

        if vector_ids:
            self.vector_store.upsert_texts(vector_ids, vector_documents, vector_metadatas)

        self.event_log.append(
            IngestEvent(
                event_type="ingest",
                source_id=source_id,
                path=str(source_path),
                payload={"title": payload["title"], "chunk_count": len(chunk_ids)},
            )
        )

        return IngestResult(
            source_id=source_id,
            chunk_ids=chunk_ids,
            node_count=1 + len(chunk_ids),
            edge_count=len(chunk_ids),
            metadata={"title": payload["title"], "kind": payload["metadata"]["kind"]},
        )

    def _extract(self, source_path: Path) -> dict[str, Any]:
        if source_path.is_dir():
            return self.git_extractor.extract(source_path)

        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            return self.pdf_extractor.extract(source_path)
        if suffix in {".md", ".txt"}:
            return self.note_extractor.extract(source_path)
        if suffix in {".json", ".yaml", ".yml"}:
            return self.experiment_extractor.extract(source_path)
        raise ValueError(f"Unsupported ingest path: {source_path}")

    def _source_kind(self, raw_kind: str) -> NodeKind:
        mapping = {
            "note": NodeKind.note,
            "pdf": NodeKind.paper,
            "experiment": NodeKind.experiment,
            "git": NodeKind.source,
        }
        return mapping.get(raw_kind, NodeKind.source)

    def _chunk_content(self, content: str) -> list[str]:
        normalized = content.strip()
        if not normalized:
            return []

        window = max(200, self.settings.chunk_size)
        overlap = min(self.settings.chunk_overlap, window // 2)
        cursor = 0
        chunks: list[str] = []

        while cursor < len(normalized):
            limit = min(len(normalized), cursor + window)
            split = limit
            if limit < len(normalized):
                for marker in ("\n\n", "\n", " "):
                    candidate = normalized.rfind(marker, cursor, limit)
                    if candidate > cursor + (window // 2):
                        split = candidate + len(marker)
                        break

            chunk = normalized[cursor:split].strip()
            if chunk:
                chunks.append(chunk)

            if split >= len(normalized):
                break

            next_cursor = max(split - overlap, cursor + 1)
            cursor = next_cursor

        return chunks
