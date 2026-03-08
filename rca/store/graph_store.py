"""SQLite-backed graph storage for nodes and edges."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rca.contracts.nodes import Edge, Node


class GraphStore:
    """Persist nodes and edges in a local SQLite database."""

    def __init__(self, db_path: str | Path, schema_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path) if schema_path else Path(__file__).with_name("migrations") / "schema.sql"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def create_schema(self) -> None:
        schema_sql = self.schema_path.read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.executescript(schema_sql)

    def upsert_node(self, node: Node) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO nodes (id, kind, title, text, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    title = excluded.title,
                    text = excluded.text,
                    metadata = excluded.metadata
                """,
                (
                    node.id,
                    node.kind.value,
                    node.title,
                    node.text,
                    json.dumps(node.metadata, sort_keys=True),
                    node.created_at.isoformat(),
                ),
            )

    def upsert_edge(self, edge: Edge) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO edges (source, target, kind, weight, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, target, kind) DO UPDATE SET
                    weight = excluded.weight,
                    metadata = excluded.metadata
                """,
                (
                    edge.source,
                    edge.target,
                    edge.kind.value,
                    edge.weight,
                    json.dumps(edge.metadata, sort_keys=True),
                    edge.created_at.isoformat(),
                ),
            )

    def get_node(self, node_id: str) -> Node | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, kind, title, text, metadata, created_at FROM nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
        if row is None:
            return None
        return Node.model_validate(
            {
                "id": row["id"],
                "kind": row["kind"],
                "title": row["title"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
            }
        )

    def list_edges(self, node_id: str) -> list[Edge]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT source, target, kind, weight, metadata, created_at
                FROM edges
                WHERE source = ? OR target = ?
                ORDER BY created_at ASC
                """,
                (node_id, node_id),
            ).fetchall()
        return [
            Edge.model_validate(
                {
                    "source": row["source"],
                    "target": row["target"],
                    "kind": row["kind"],
                    "weight": row["weight"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                }
            )
            for row in rows
        ]

    def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
        token = f"%{query.lower()}%"
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, kind, title, text, metadata, created_at
                FROM nodes
                WHERE lower(title) LIKE ? OR lower(coalesce(text, '')) LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (token, token, limit),
            ).fetchall()
        return [
            Node.model_validate(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "title": row["title"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                }
            )
            for row in rows
        ]
