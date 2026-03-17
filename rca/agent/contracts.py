"""Agent loop data contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ToolCallStatus(StrEnum):
    success = "success"
    error = "error"


class ToolCallTrace(BaseModel):
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: str
    status: ToolCallStatus
    duration_ms: float


class AgentTrace(BaseModel):
    query: str
    timestamp: str = Field(default_factory=_now)
    iterations: int = 0
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    stopped_reason: str = ""
    warnings: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    query: str
    answer: str
    trace: AgentTrace
    error: str | None = None
