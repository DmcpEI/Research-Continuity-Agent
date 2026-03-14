"""Trace contracts for query observability."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StageTrace(BaseModel):
    name: str
    duration_ms: float
    hit_count: int
    notes: str = ""


class HitProvenance(BaseModel):
    node_id: str
    score: float
    stage: str
    rank: int


class QueryTrace(BaseModel):
    query: str
    rewritten_query: str | None = None
    timestamp: str = Field(default_factory=_iso_utc_now)
    stages: list[StageTrace] = Field(default_factory=list)
    provenance: list[HitProvenance] = Field(default_factory=list)
    context_node_ids: list[str] = Field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    corpus_version: str = ""
