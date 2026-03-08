"""Append-only event log for ingest activity."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class IngestEvent(BaseModel):
    """A single append-only event emitted during ingest."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: str
    source_id: str
    path: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)


class EventLog:
    """Persist ingest events to a JSONL file."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.touch(exist_ok=True)

    def append(self, event: IngestEvent) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")

    def list_recent(self, limit: int = 50) -> list[IngestEvent]:
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        selected = lines[-limit:]
        return [IngestEvent.model_validate_json(line) for line in selected if line.strip()]
