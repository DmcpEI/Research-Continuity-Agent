from __future__ import annotations

from pathlib import Path

import pytest

from rca.config.settings import Settings
from rca.contracts.ids import make_source_id
from rca.contracts.nodes import Node, NodeKind
from rca.flows.ingest_flow import IngestFlow
from rca.flows.retrieve_flow import RetrieveFlow
from rca.store.vector_store import VectorStore


def _settings_for(tmp_path: Path) -> Settings:
    runtime_dir = tmp_path / ".rca"
    return Settings(
        data_dir=runtime_dir,
        graph_db_path=runtime_dir / "graph.sqlite3",
        vector_dir=runtime_dir / "vectors",
        event_log_path=runtime_dir / "events.jsonl",
        telemetry_log_path=runtime_dir / "telemetry.jsonl",
        experiment_db_path=runtime_dir / "experiments.sqlite3",
        tool_policy_path=Path("rca/config/tool_policies.yaml"),
        # Real cross-encoder loads hit the Hugging Face Hub (slow, network-dependent);
        # reranking is covered with a stub in tests/unit/test_retrieve_flow.py.
        enable_reranker=False,
    )


def test_ingest_flow_indexes_markdown_note(tmp_path) -> None:
    settings = _settings_for(tmp_path)
    note_path = tmp_path / "demo-note.md"
    note_path.write_text(
        "# Demo Note\n\nThis repo tracks research continuity across experiments.\n",
        encoding="utf-8",
    )

    ingest_flow = IngestFlow(settings=settings)
    result = ingest_flow.ingest_path(note_path)

    retrieve_flow = RetrieveFlow(
        settings=settings,
        graph_store=ingest_flow.graph_store,
        vector_store=ingest_flow.vector_store,
    )
    bundle = retrieve_flow.retrieve("research continuity")

    assert result.source_id == "src:note/demo-note"
    assert result.ingest_status == "created"
    assert result.revision_id is not None
    assert result.chunk_ids
    assert bundle.hits
    assert bundle.hits[0].metadata.get("source_id") == result.source_id


def test_ingest_flow_truncates_source_text_but_keeps_full_chunk_text(tmp_path) -> None:
    settings = _settings_for(tmp_path)
    repeated_line = (
        "Research continuity requires durable long-form source retention for chunking.\n"
    )
    long_body = "# Long Note\n\n" + (repeated_line * 80)
    note_path = tmp_path / "long-note.md"
    note_path.write_text(long_body, encoding="utf-8")

    ingest_flow = IngestFlow(settings=settings)
    result = ingest_flow.ingest_path(note_path)
    source_node = ingest_flow.graph_store.get_node(result.source_id)
    first_chunk = ingest_flow.graph_store.get_node(result.chunk_ids[0])

    assert source_node is not None
    assert first_chunk is not None
    assert source_node.text is not None
    assert len(source_node.text) < len(long_body.strip())
    assert len(source_node.text) <= max(1000, settings.chunk_size)
    assert first_chunk.text is not None
    assert len(first_chunk.text) > len(source_node.text) // 2


def test_ingest_flow_reingest_same_path_is_unchanged_without_extra_revision(tmp_path) -> None:
    settings = _settings_for(tmp_path)
    note_path = tmp_path / "same-note.md"
    note_path.write_text("# Same Note\n\nalpha beta gamma\n", encoding="utf-8")

    ingest_flow = IngestFlow(settings=settings)
    first = ingest_flow.ingest_path(note_path)
    second = ingest_flow.ingest_path(note_path)
    revisions = ingest_flow.graph_store.list_source_revisions(first.source_id)

    assert second.source_id == first.source_id
    assert second.ingest_status == "unchanged"
    assert second.revision_id == first.revision_id
    assert len(revisions) == 1


def test_ingest_flow_reingest_renamed_same_content_reuses_source_and_records_alias(
    tmp_path,
) -> None:
    settings = _settings_for(tmp_path)
    original = tmp_path / "demo-note.md"
    renamed = tmp_path / "renamed-note.md"
    content = "# Shared Title\n\nresearch continuity survives renames\n"
    original.write_text(content, encoding="utf-8")
    renamed.write_text(content, encoding="utf-8")

    ingest_flow = IngestFlow(settings=settings)
    first = ingest_flow.ingest_path(original)
    second = ingest_flow.ingest_path(renamed)
    revisions = ingest_flow.graph_store.list_source_revisions(first.source_id)

    assert second.source_id == first.source_id
    assert second.ingest_status == "unchanged"
    assert second.revision_id != first.revision_id
    assert len(revisions) == 2
    assert revisions[-1].path == str(renamed)
    assert revisions[-1].ingest_name == "renamed-note"


def test_ingest_flow_reingest_changed_content_replaces_active_chunks(tmp_path) -> None:
    settings = _settings_for(tmp_path)
    note_path = tmp_path / "versioned-note.md"
    long_line = "research continuity " * 120
    note_path.write_text(f"# Versioned\n\n{long_line}\n", encoding="utf-8")

    ingest_flow = IngestFlow(settings=settings)
    first = ingest_flow.ingest_path(note_path)
    old_chunk_ids = list(first.chunk_ids)

    note_path.write_text("# Versioned\n\nshort update\n", encoding="utf-8")
    second = ingest_flow.ingest_path(note_path)
    revisions = ingest_flow.graph_store.list_source_revisions(first.source_id)
    vector_store = VectorStore(settings.vector_dir, settings.default_collection)

    assert second.source_id == first.source_id
    assert second.ingest_status == "updated"
    assert second.revision_id != first.revision_id
    assert len(second.chunk_ids) < len(old_chunk_ids)
    assert len(revisions) == 2
    assert revisions[-1].content_sha256 != revisions[0].content_sha256
    for stale_chunk_id in old_chunk_ids[len(second.chunk_ids) :]:
        assert ingest_flow.graph_store.get_node(stale_chunk_id) is None
        assert stale_chunk_id not in vector_store._documents


def test_ingest_flow_legacy_source_node_is_detected_as_updated(tmp_path) -> None:
    settings = _settings_for(tmp_path)
    note_path = tmp_path / "legacy-note.md"
    note_path.write_text("# Legacy\n\noriginal content\n", encoding="utf-8")

    ingest_flow = IngestFlow(settings=settings)
    legacy_source_id = make_source_id("note", "legacy-note")
    ingest_flow.graph_store.upsert_node(
        Node(id=legacy_source_id, kind=NodeKind.note, title="Legacy", text="original content")
    )

    result = ingest_flow.ingest_path(note_path)

    assert result.source_id == legacy_source_id
    assert result.ingest_status == "updated"
    assert result.revision_id is not None
    revisions = ingest_flow.graph_store.list_source_revisions(legacy_source_id)
    assert len(revisions) == 1
    assert revisions[0].ingest_status == "updated"
    assert revisions[0].metadata["matched_by"] == "legacy_source_id"


def test_ingest_flow_partial_failure_self_heals_on_retry(tmp_path) -> None:
    settings = _settings_for(tmp_path)
    note_path = tmp_path / "self-heal-note.md"
    note_path.write_text("# Heal\n\noriginal\n", encoding="utf-8")

    ingest_flow = IngestFlow(settings=settings)
    first = ingest_flow.ingest_path(note_path)
    assert first.ingest_status == "created"

    note_path.write_text("# Heal\n\nupdated content\n", encoding="utf-8")
    original_upsert = ingest_flow.vector_store.upsert_texts

    call_count = 0

    def failing_upsert(ids, documents, metadatas, embeddings=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated vector write failure")
        original_upsert(ids, documents, metadatas, embeddings)

    ingest_flow.vector_store.upsert_texts = failing_upsert

    with pytest.raises(RuntimeError, match="simulated vector write failure"):
        ingest_flow.ingest_path(note_path)

    revisions_after_failure = ingest_flow.graph_store.list_source_revisions(first.source_id)
    assert len(revisions_after_failure) == 1

    ingest_flow.vector_store.upsert_texts = original_upsert
    recovery = ingest_flow.ingest_path(note_path)

    assert recovery.ingest_status == "updated"
    assert recovery.revision_id != first.revision_id
    revisions_after_recovery = ingest_flow.graph_store.list_source_revisions(first.source_id)
    assert len(revisions_after_recovery) == 2
