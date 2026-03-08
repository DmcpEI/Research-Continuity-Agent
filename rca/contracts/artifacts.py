"""Artifact DTOs projected from sources and graph records."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from rca.contracts.ids import SourceID


class ArtifactType(str, Enum):
    paper = "paper"
    note = "note"
    experiment = "experiment"


class Artifact(BaseModel):
    """Base artifact shared by papers, notes, and experiments."""

    artifact_id: SourceID
    artifact_type: ArtifactType
    title: str
    summary: str | None = None
    source_ids: list[SourceID] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperArtifact(Artifact):
    artifact_type: Literal[ArtifactType.paper] = ArtifactType.paper
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None


class NoteArtifact(Artifact):
    artifact_type: Literal[ArtifactType.note] = ArtifactType.note
    note_path: str | None = None


class ExperimentArtifact(Artifact):
    artifact_type: Literal[ArtifactType.experiment] = ArtifactType.experiment
    status: str = "unknown"
    metrics: dict[str, float] = Field(default_factory=dict)
