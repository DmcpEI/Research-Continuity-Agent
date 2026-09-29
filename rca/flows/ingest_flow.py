"""Business logic for ingesting artifacts into local stores."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rca.config.settings import Settings, get_settings
from rca.contracts.ids import make_chunk_id, make_source_id, normalize_identifier
from rca.contracts.nodes import Edge, EdgeKind, Node, NodeKind
from rca.contracts.revisions import IngestStatus, SourceRevision
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
    revision_id: str | None = None
    ingest_status: IngestStatus = IngestStatus.created
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class ResolvedSource:
    source_id: str
    ingest_status: IngestStatus
    matched_by: str
    matched_revision: SourceRevision | None = None
    record_revision: bool = True


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
        self.vector_store = vector_store or VectorStore(
            self.settings.vector_dir, self.settings.default_collection
        )
        self.event_log = event_log or EventLog(self.settings.event_log_path)
        self.note_extractor = note_extractor or NoteExtractor()
        self.pdf_extractor = pdf_extractor or PDFExtractor()
        self.git_extractor = git_extractor or GitExtractor()
        self.experiment_extractor = experiment_extractor or ExperimentExtractor()

    def ingest_path(self, path: str | Path) -> IngestResult:
        source_path = Path(path)
        payload = self._extract(source_path)
        ingest_name = self._normalize_ingest_name(source_path)
        file_sha256 = self._file_sha256(source_path)
        content_sha256 = self._content_sha256(payload["content"])
        resolution = self._resolve_source(
            source_path=source_path,
            kind=payload["metadata"]["kind"],
            ingest_name=ingest_name,
            file_sha256=file_sha256,
            content_sha256=content_sha256,
        )
        source_id = resolution.source_id
        source_kind = self._source_kind(payload["metadata"]["kind"])
        existing_source = self.graph_store.get_node(source_id)
        pending_revision = self._build_pending_revision(
            source_id=source_id,
            source_path=source_path,
            payload=payload,
            ingest_name=ingest_name,
            file_sha256=file_sha256,
            content_sha256=content_sha256,
            resolution=resolution,
        )
        active_revision = pending_revision or resolution.matched_revision
        revision_id = active_revision.revision_id if active_revision is not None else None

        chunk_ids: list[str] = []
        if resolution.ingest_status is not IngestStatus.unchanged:
            source_preview = self._source_text_preview(payload["content"])
            source_node = Node(
                id=source_id,
                kind=source_kind,
                title=self._resolve_source_title(payload, existing_source),
                text=source_preview or None,
                metadata=self._merge_source_metadata(
                    payload=payload,
                    existing_source=existing_source,
                    revision_id=revision_id,
                    file_sha256=file_sha256,
                    content_sha256=content_sha256,
                    ingest_name=ingest_name,
                    source_path=source_path,
                ),
            )

            chunk_nodes: list[Node] = []
            chunk_edges: list[Edge] = []
            vector_ids: list[str] = []
            vector_documents: list[str] = []
            vector_metadatas: list[dict[str, Any]] = []

            for index, chunk_text in enumerate(self._chunk_content(payload["content"])):
                chunk_id = make_chunk_id(source_id, index)
                chunk_ids.append(chunk_id)

                chunk_nodes.append(
                    Node(
                        id=chunk_id,
                        kind=NodeKind.chunk,
                        title=f"{source_node.title} #{index + 1}",
                        text=chunk_text,
                        metadata={
                            "source_id": source_id,
                            "chunk_index": index,
                            "kind": payload["metadata"]["kind"],
                            "revision_id": revision_id,
                        },
                    )
                )

                chunk_edges.append(
                    Edge(
                        source=source_id,
                        target=chunk_id,
                        kind=EdgeKind.contains,
                        metadata={"sequence": index, "revision_id": revision_id},
                    )
                )

                vector_ids.append(chunk_id)
                vector_documents.append(chunk_text)
                vector_metadatas.append(
                    {
                        "source_id": source_id,
                        "title": source_node.title,
                        "revision_id": revision_id,
                    }
                )

            old_chunk_ids = self.graph_store.replace_source_chunks_atomic(
                source_id=source_id,
                source_node=source_node,
                chunk_nodes=chunk_nodes,
                chunk_edges=chunk_edges,
            )

            if old_chunk_ids:
                self.vector_store.delete_texts(old_chunk_ids)

            if vector_ids:
                self.vector_store.upsert_texts(vector_ids, vector_documents, vector_metadatas)
        elif pending_revision is not None and existing_source is not None:
            self.graph_store.upsert_node(
                Node(
                    id=existing_source.id,
                    kind=existing_source.kind,
                    title=existing_source.title,
                    text=existing_source.text,
                    metadata=self._merge_source_metadata(
                        payload=payload,
                        existing_source=existing_source,
                        revision_id=revision_id,
                        file_sha256=file_sha256,
                        content_sha256=content_sha256,
                        ingest_name=ingest_name,
                        source_path=source_path,
                    ),
                    created_at=existing_source.created_at,
                )
            )
            chunk_ids = self.graph_store.list_source_chunk_ids(source_id)
        else:
            chunk_ids = self.graph_store.list_source_chunk_ids(source_id)

        if pending_revision is not None:
            self.graph_store.insert_source_revision(pending_revision)

        self.event_log.append(
            IngestEvent(
                event_type="ingest",
                source_id=source_id,
                path=str(source_path),
                payload={
                    "title": payload["title"],
                    "chunk_count": len(chunk_ids),
                    "revision_id": revision_id,
                    "ingest_status": resolution.ingest_status.value,
                    "matched_by": resolution.matched_by,
                },
            )
        )

        return IngestResult(
            source_id=source_id,
            chunk_ids=chunk_ids,
            node_count=1 + len(chunk_ids),
            edge_count=len(chunk_ids),
            revision_id=revision_id,
            ingest_status=resolution.ingest_status,
            metadata={
                "title": self._resolve_source_title(payload, existing_source),
                "kind": payload["metadata"]["kind"],
            },
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

    def _build_pending_revision(
        self,
        source_id: str,
        source_path: Path,
        payload: dict[str, Any],
        ingest_name: str,
        file_sha256: str | None,
        content_sha256: str,
        resolution: ResolvedSource,
    ) -> SourceRevision | None:
        if not resolution.record_revision:
            return None

        return SourceRevision(
            source_id=source_id,
            revision_number=self.graph_store.next_revision_number(source_id),
            ingest_status=resolution.ingest_status,
            path=str(source_path),
            ingest_name=ingest_name,
            title=payload["title"],
            file_sha256=file_sha256,
            content_sha256=content_sha256,
            metadata={"matched_by": resolution.matched_by, **payload["metadata"]},
        )

    def _resolve_source(
        self,
        source_path: Path,
        kind: str,
        ingest_name: str,
        file_sha256: str | None,
        content_sha256: str,
    ) -> ResolvedSource:
        source_path_str = str(source_path)

        file_match = self.graph_store.find_revision_by_file_hash(file_sha256)
        if file_match is not None:
            should_record = (
                file_match.path != source_path_str or file_match.ingest_name != ingest_name
            )
            return ResolvedSource(
                source_id=file_match.source_id,
                ingest_status=IngestStatus.unchanged,
                matched_by="file_hash",
                matched_revision=file_match,
                record_revision=should_record,
            )

        content_match = self.graph_store.find_revision_by_content_hash(content_sha256)
        if content_match is not None:
            return ResolvedSource(
                source_id=content_match.source_id,
                ingest_status=IngestStatus.unchanged,
                matched_by="content_hash",
                matched_revision=content_match,
                record_revision=True,
            )

        path_match = self.graph_store.find_revision_by_path(source_path_str)
        if path_match is not None:
            return ResolvedSource(
                source_id=path_match.source_id,
                ingest_status=IngestStatus.updated,
                matched_by="path",
                matched_revision=path_match,
                record_revision=True,
            )

        name_match = self.graph_store.find_revision_by_ingest_name(ingest_name)
        if name_match is not None:
            return ResolvedSource(
                source_id=name_match.source_id,
                ingest_status=IngestStatus.updated,
                matched_by="ingest_name",
                matched_revision=name_match,
                record_revision=True,
            )

        candidate_source_id = make_source_id(kind, ingest_name)
        if self.graph_store.get_node(candidate_source_id) is not None:
            return ResolvedSource(
                source_id=candidate_source_id,
                ingest_status=IngestStatus.updated,
                matched_by="legacy_source_id",
                record_revision=True,
            )

        return ResolvedSource(
            source_id=candidate_source_id,
            ingest_status=IngestStatus.created,
            matched_by="new_source",
            record_revision=True,
        )

    @staticmethod
    def _normalize_ingest_name(source_path: Path) -> str:
        raw_name = source_path.stem if source_path.is_file() else source_path.name
        normalized = normalize_identifier(raw_name)
        return normalized or "source"

    @staticmethod
    def _normalize_content(content: str) -> str:
        return re.sub(r"\s+", " ", content).strip()

    def _content_sha256(self, content: str) -> str:
        normalized = self._normalize_content(content)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_sha256(source_path: Path) -> str | None:
        if not source_path.is_file():
            return None

        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _resolve_source_title(payload: dict[str, Any], existing_source: Node | None) -> str:
        if existing_source is not None and payload["metadata"].get("kind") == "pdf":
            return existing_source.title
        return payload["title"]

    @staticmethod
    def _merge_source_metadata(
        payload: dict[str, Any],
        existing_source: Node | None,
        revision_id: str | None,
        file_sha256: str | None,
        content_sha256: str,
        ingest_name: str,
        source_path: Path,
    ) -> dict[str, Any]:
        metadata = dict(existing_source.metadata) if existing_source is not None else {}
        metadata.update(payload["metadata"])
        metadata.update(
            {
                "current_revision_id": revision_id,
                "file_sha256": file_sha256,
                "content_sha256": content_sha256,
                "latest_path": str(source_path),
                "ingest_name": ingest_name,
            }
        )
        return metadata

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

    def _source_text_preview(self, content: str) -> str:
        normalized = content.strip()
        if not normalized:
            return ""

        limit = max(1000, self.settings.chunk_size)
        if len(normalized) <= limit:
            return normalized

        for marker in ("\n\n", "\n", " "):
            candidate = normalized.rfind(marker, 0, limit)
            if candidate >= max(200, limit // 2):
                return normalized[:candidate].strip()

        return normalized[:limit].strip()
