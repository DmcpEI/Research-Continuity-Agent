from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from rca.contracts.ids import ChunkID, SourceID, make_chunk_id, make_source_id


class IdentifierEnvelope(BaseModel):
    source_id: SourceID
    chunk_id: ChunkID


def test_identifier_helpers_create_valid_values() -> None:
    source_id = make_source_id("note", "Demo Note")
    chunk_id = make_chunk_id(source_id, 0)

    model = IdentifierEnvelope(source_id=source_id, chunk_id=chunk_id)

    assert model.source_id == "src:note/demo-note"
    assert model.chunk_id == "chk:note/demo-note:0000"


def test_identifier_model_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        IdentifierEnvelope(source_id="bad-id", chunk_id="also-bad")
