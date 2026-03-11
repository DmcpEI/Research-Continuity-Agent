"""Demo experiment server backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from pydantic import BaseModel, Field


class ExperimentRecord(BaseModel):
    run_id: str
    name: str
    status: Literal["pending", "running", "complete", "failed"]
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
    
    def update_run(
        self,
        run_id: str,
        status: Literal["pending", "running", "complete", "failed"] | None = None,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """Update an existing experiment run."""
        # First fetch the existing record
        existing = self.get_run(run_id)
        if existing is None:
            raise ValueError(f"No experiment found with run_id: {run_id}")

        # Merge — only overwrite fields that were explicitly passed
        new_status = status if status is not None else existing.status
        new_metrics = {**existing.metrics, **(metrics or {})}
        new_metadata = {**existing.metadata, **(metadata or {})}
        updated_at = datetime.now(timezone.utc).isoformat()

        with self.connect() as connection:
            connection.execute(
                """
                UPDATE experiments
                SET status = ?, metrics = ?, metadata = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    new_status,
                    json.dumps(new_metrics, sort_keys=True),
                    json.dumps(new_metadata, sort_keys=True),
                    updated_at,
                    run_id,
                ),
            )

        # Return the updated record
        return self.get_run(run_id)


    def get_run(self, run_id: str) -> ExperimentRecord | None:
        """Fetch a single experiment by run_id."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return ExperimentRecord.model_validate({
            "run_id": row["run_id"],
            "name": row["name"],
            "status": row["status"],
            "metrics": json.loads(row["metrics"]),
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
        })
    
# --- MCP Server setup ---

def make_server(db_path: str) -> Server:
    exp = ExperimentServer(db_path)
    server = Server("experiments")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="record_run",
                description="Record a new experiment run with name, status, metrics, and metadata",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "running", "complete", "failed"],
                            "default": "pending",
                        },
                        "metrics": {"type": "object", "additionalProperties": {"type": "number"}},
                        "metadata": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["name"],
                },
            ),
            types.Tool(
                name="list_runs",
                description="List recent experiment runs, optionally limited by a count",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 20},
                    },
                },
            ),
            types.Tool(
                name="update_run",
                description="Update an existing experiment run's status, metrics, or metadata",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "running", "complete", "failed"],
                        },
                        "metrics": {"type": "object", "additionalProperties": {"type": "number"}},
                        "metadata": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["run_id"],
                },
            ),
            types.Tool(
                name="get_run",
                description="Fetch a single experiment run by its run_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                    },
                    "required": ["run_id"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "record_run":
                result = exp.record_run(**arguments)
                text = f"Run recorded: {result.run_id}"
            elif name == "list_runs":
                runs = exp.list_runs(limit=arguments.get("limit", 20))
                text = "\n".join(
                    f"{r.run_id} | {r.name} | {r.status} | {r.created_at}"
                    for r in runs
                ) or "No runs found."
            elif name == "update_run":
                result = exp.update_run(**arguments)
                text = f"Run updated: {result.run_id}"
            elif name == "get_run":
                result = exp.get_run(arguments["run_id"])
                if result is None:
                    text = "No run found with that run_id."
                else:
                    text = (
                        f"Run ID: {result.run_id}\n"
                        f"Name: {result.name}\n"
                        f"Status: {result.status}\n"
                        f"Metrics: {json.dumps(result.metrics, indent=2)}\n"
                        f"Metadata: {json.dumps(result.metadata, indent=2)}\n"
                        f"Created At: {result.created_at}"
                    )
            else:
                text = f"Unknown tool: {name}"
        except (ValueError, Exception) as e:
            text = f"Error: {e}"
        return [types.TextContent(type="text", text=text)]

    return server

async def main(db_path: str) -> None:
    server = make_server(db_path)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio, sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else ".rca/experiments.sqlite3"
    asyncio.run(main(db_path))
