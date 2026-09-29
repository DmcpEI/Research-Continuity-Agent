"""Contracts for document revision history."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from rca.contracts.ids import SourceID


class IngestStatus(StrEnum):
    created = "created"
    unchanged = "unchanged"
    updated = "updated"


class SourceRevision(BaseModel):
    """Append-only revision record for a logical source."""

    revision_id: str = Field(default_factory=lambda: uuid4().hex)
    source_id: SourceID
    revision_number: int
    ingest_status: IngestStatus
    path: str
    ingest_name: str
    title: str
    file_sha256: str | None = None
    content_sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
