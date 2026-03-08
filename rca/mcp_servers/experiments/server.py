"""Demo experiment server backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ExperimentRecord(BaseModel):
    run_id: str
    name: str
    status: str
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ExperimentServer:
    """Persist experiment runs in a local SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.schema_path = Path(__file__).with_name("schema.sql")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def create_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(self.schema_path.read_text(encoding="utf-8"))

    def record_run(
        self,
        name: str,
        status: str = "pending",
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> ExperimentRecord:
        record = ExperimentRecord(
            run_id=run_id or uuid4().hex,
            name=name,
            status=status,
            metrics=metrics or {},
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
        )
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments (run_id, name, status, metrics, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.name,
                    record.status,
                    json.dumps(record.metrics, sort_keys=True),
                    json.dumps(record.metadata, sort_keys=True),
                    record.created_at.isoformat(),
                ),
            )
        return record

    def list_runs(self, limit: int = 20) -> list[ExperimentRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, name, status, metrics, metadata, created_at
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            ExperimentRecord.model_validate(
                {
                    "run_id": row["run_id"],
                    "name": row["name"],
                    "status": row["status"],
                    "metrics": json.loads(row["metrics"]),
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                }
            )
            for row in rows
        ]
