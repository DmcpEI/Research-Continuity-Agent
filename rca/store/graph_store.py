"""SQLite-backed graph storage for nodes and edges."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from rca.contracts.nodes import Edge, Node

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9+_.-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}


class GraphStore:
    """Persist nodes and edges in a local SQLite database."""

    def __init__(self, db_path: str | Path, schema_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.schema_path = (
            Path(schema_path)
            if schema_path
            else Path(__file__).with_name("migrations") / "schema.sql"
        )
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
            self._sync_fts_index(connection)

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

    @staticmethod
    def _escape_like(token: str) -> str:
        """Escape SQL LIKE special characters so _ and % are treated as literals."""
        return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        return [
            token
            for token in TOKEN_PATTERN.findall(query.lower())
            if len(token) >= 3 and token not in STOPWORDS
        ]

    @staticmethod
    def _rows_to_nodes(rows: list[sqlite3.Row]) -> list[Node]:
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

    @staticmethod
    def _sync_fts_index(connection: sqlite3.Connection) -> None:
        # Rebuild is idempotent and ensures older databases created before the
        # FTS5 table/triggers existed are fully indexed for BM25 baseline runs.
        connection.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")

    def search_nodes_like(self, query: str, limit: int = 10) -> list[Node]:
        tokens = self._tokenize_query(query)
        if not tokens:
            tokens = [query.lower().strip()]

        score_clauses = []
        score_params: list[str] = []
        where_clauses = []
        where_params: list[str] = []

        for token in tokens:
            pattern = f"%{self._escape_like(token)}%"
            score_clauses.append(
                "(CASE WHEN lower(title) LIKE ? ESCAPE '\\' THEN 3 ELSE 0 END + "
                "CASE WHEN lower(coalesce(text, '')) LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END)"
            )
            score_params.extend([pattern, pattern])
            where_clauses.append(
                "(lower(title) LIKE ? ESCAPE '\\' OR lower(coalesce(text, '')) LIKE ? ESCAPE '\\')"
            )
            where_params.extend([pattern, pattern])

        score_sql = " + ".join(score_clauses)
        where_sql = " OR ".join(where_clauses)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, kind, title, text, metadata, created_at
                FROM nodes
                WHERE {where_sql}
                ORDER BY {score_sql} DESC, created_at DESC
                LIMIT ?
                """,
                [*where_params, *score_params, limit],
            ).fetchall()
        return self._rows_to_nodes(rows)

    def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
        tokens = self._tokenize_query(query)
        if not tokens:
            stripped = query.strip()
            if not stripped:
                return []
            tokens = [stripped.lower()]

        match_query = " OR ".join(f'"{token}"' for token in tokens)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT n.id, n.kind, n.title, n.text, n.metadata, n.created_at
                FROM nodes_fts
                JOIN nodes AS n ON n.rowid = nodes_fts.rowid
                WHERE nodes_fts MATCH ?
                ORDER BY bm25(nodes_fts), n.created_at DESC
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
        return self._rows_to_nodes(rows)
