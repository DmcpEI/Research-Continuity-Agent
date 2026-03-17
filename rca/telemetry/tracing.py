"""Tracing helpers for runs, prompts, and tool calls."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolCallEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    tool_name: str
    payload: dict[str, Any] = Field(default_factory=dict)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class Tracer:
    """Write simple JSONL traces to a local file."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.touch(exist_ok=True)

    @staticmethod
    def new_run_id() -> str:
        return uuid4().hex

    def log_tool_call(
        self, run_id: str, tool_name: str, payload: dict[str, Any] | None = None
    ) -> None:
        event = ToolCallEvent(run_id=run_id, tool_name=tool_name, payload=payload or {})
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")
