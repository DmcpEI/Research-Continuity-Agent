"""Stable identifier rules shared across the system."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import StringConstraints

SOURCE_ID_PATTERN = r"^src:[a-z0-9][a-z0-9._/-]{2,127}$"
CHUNK_ID_PATTERN = r"^chk:[a-z0-9][a-z0-9._/-]{2,127}:[0-9]{4}$"
NODE_ID_PATTERN = r"^(?:src|chk):[a-z0-9][a-z0-9._/-]{2,127}(?::[0-9]{4})?$"

SourceID = Annotated[str, StringConstraints(pattern=SOURCE_ID_PATTERN)]
ChunkID = Annotated[str, StringConstraints(pattern=CHUNK_ID_PATTERN)]
NodeID = Annotated[str, StringConstraints(pattern=NODE_ID_PATTERN)]

_INVALID_IDENTIFIER_CHARS = re.compile(r"[^a-z0-9._/-]+")
_DUPLICATE_SEPARATORS = re.compile(r"[./_-]{2,}")


def normalize_identifier(value: str) -> str:
    """Convert human input into a lower-case identifier fragment."""

    normalized = value.strip().lower()
    normalized = normalized.replace(" ", "-")
    normalized = _INVALID_IDENTIFIER_CHARS.sub("-", normalized)
    normalized = _DUPLICATE_SEPARATORS.sub("-", normalized)
    return normalized.strip("./_-")


def make_source_id(namespace: str, name: str) -> str:
    """Build a stable source ID from a namespace and local name."""

    namespace_part = normalize_identifier(namespace)
    name_part = normalize_identifier(name)
    if namespace_part:
        return f"src:{namespace_part}/{name_part}"
    return f"src:{name_part}"


def make_chunk_id(source_id: str, index: int) -> str:
    """Derive a chunk ID from a source ID and zero-based chunk index."""

    if not is_source_id(source_id):
        raise ValueError(f"Invalid source ID: {source_id}")
    if index < 0:
        raise ValueError("Chunk index must be non-negative.")
    return f"chk:{source_id[4:]}:{index:04d}"


def is_source_id(value: str) -> bool:
    """Return whether the value matches the source ID contract."""

    return re.fullmatch(SOURCE_ID_PATTERN, value) is not None


def is_chunk_id(value: str) -> bool:
    """Return whether the value matches the chunk ID contract."""

    return re.fullmatch(CHUNK_ID_PATTERN, value) is not None
