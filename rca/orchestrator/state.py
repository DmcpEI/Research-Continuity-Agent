"""Typed state object for orchestration."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class OrchestratorState(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    user_query: str | None = None
    route: str | None = None
    requested_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    retrieved_hits: list[dict[str, Any]] = Field(default_factory=list)
    final_output: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
