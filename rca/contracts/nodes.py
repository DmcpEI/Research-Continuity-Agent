"""Graph node and edge models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rca.contracts.ids import NodeID


class NodeKind(str, Enum):
    source = "source"
    chunk = "chunk"
    note = "note"
    paper = "paper"
    experiment = "experiment"
    digest = "digest"


class EdgeKind(str, Enum):
    contains = "contains"
    derived_from = "derived_from"
    references = "references"
    cites = "cites"
    related_to = "related_to"
    produced_by = "produced_by"


class Node(BaseModel):
    """Graph node stored in SQLite."""

    id: NodeID
    kind: NodeKind
    title: str
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Edge(BaseModel):
    """Directed relationship between nodes."""

    source: NodeID
    target: NodeID
    kind: EdgeKind
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
