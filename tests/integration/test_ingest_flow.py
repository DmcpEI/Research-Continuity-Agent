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
        telemetry_log_path=runtime_dir / "telemetry.jsonl",
        experiment_db_path=runtime_dir / "experiments.sqlite3",
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
    # The graph holds the new chunks but the commit record (the revision row) is missing.
    assert ingest_flow.graph_store.source_is_incomplete(first.source_id) is True

    ingest_flow.vector_store.upsert_texts = original_upsert
    recovery = ingest_flow.ingest_path(note_path)

    assert recovery.ingest_status == "updated"
    assert recovery.revision_id != first.revision_id
    revisions_after_recovery = ingest_flow.graph_store.list_source_revisions(first.source_id)
    assert len(revisions_after_recovery) == 2
    assert ingest_flow.graph_store.source_is_incomplete(first.source_id) is False


def test_ingest_flow_failed_first_ingest_is_incomplete_then_converges(tmp_path) -> None:
    settings = _settings_for(tmp_path)
    note_path = tmp_path / "new-note.md"
    note_path.write_text("# New\n\nfirst content\n", encoding="utf-8")
    ingest_flow = IngestFlow(settings=settings)
    original_upsert = ingest_flow.vector_store.upsert_texts

    def failing_upsert(ids, documents, metadatas, embeddings=None):
        raise RuntimeError("simulated vector write failure")

    ingest_flow.vector_store.upsert_texts = failing_upsert
    try:
        with pytest.raises(RuntimeError, match="simulated vector write failure"):
            ingest_flow.ingest_path(note_path)
    finally:
        ingest_flow.vector_store.upsert_texts = original_upsert

    source_id = make_source_id("note", "new-note")
    assert ingest_flow.graph_store.get_node(source_id) is not None
    assert ingest_flow.graph_store.list_source_revisions(source_id) == []
    assert ingest_flow.graph_store.source_is_incomplete(source_id) is True

    retry = ingest_flow.ingest_path(note_path)

    assert retry.source_id == source_id
    assert len(ingest_flow.graph_store.list_source_revisions(source_id)) == 1
    assert ingest_flow.graph_store.source_is_incomplete(source_id) is False
    assert {hit.id for hit in ingest_flow.vector_store.query("first content", 5)} >= set(
        retry.chunk_ids
    )


def test_an_incomplete_source_is_rewritten_even_when_its_file_is_unchanged(tmp_path) -> None:
    """A failed v2 update, then v1 again: v1's file has a committed revision, but the graph
    holds v2's chunks, so "unchanged" would leave them and the flag in place."""
    settings = _settings_for(tmp_path)
    note_path = tmp_path / "note.md"
    note_path.write_text("# Note\n\nversion one text\n", encoding="utf-8")
    ingest_flow = IngestFlow(settings=settings)
    first = ingest_flow.ingest_path(note_path)

    note_path.write_text("# Note\n\nversion two text\n", encoding="utf-8")
    original_upsert = ingest_flow.vector_store.upsert_texts

    def failing_upsert(ids, documents, metadatas, embeddings=None):
        raise RuntimeError("simulated vector write failure")

    ingest_flow.vector_store.upsert_texts = failing_upsert
    try:
        with pytest.raises(RuntimeError):
            ingest_flow.ingest_path(note_path)
    finally:
        ingest_flow.vector_store.upsert_texts = original_upsert
    assert ingest_flow.graph_store.source_is_incomplete(first.source_id) is True

    note_path.write_text("# Note\n\nversion one text\n", encoding="utf-8")
    again = ingest_flow.ingest_path(note_path)

    assert again.ingest_status == "updated"
    assert ingest_flow.graph_store.source_is_incomplete(first.source_id) is False
    chunk = ingest_flow.graph_store.get_node(again.chunk_ids[0])
    assert "version one" in chunk.text


def test_source_is_incomplete_is_false_for_sources_from_before_revisions(tmp_path) -> None:
    ingest_flow = IngestFlow(settings=_settings_for(tmp_path))
    ingest_flow.graph_store.upsert_node(
        Node(id="src:pdf/legacy", kind=NodeKind.paper, title="Legacy", metadata={"kind": "pdf"})
    )

    assert ingest_flow.graph_store.source_is_incomplete("src:pdf/legacy") is False
    assert ingest_flow.graph_store.source_is_incomplete("src:pdf/missing") is False


def test_ingest_writes_no_event_log(tmp_path, monkeypatch) -> None:
    """The ingest event log was write-only and was removed; ingest must not create one."""
    monkeypatch.chdir(tmp_path)  # any default relative path lands in tmp, never in ./.rca
    note_path = tmp_path / "note.md"
    note_path.write_text("# Note\n\nResearch continuity.\n", encoding="utf-8")

    IngestFlow(settings=_settings_for(tmp_path)).ingest_path(note_path)

    assert not list(tmp_path.rglob("events.jsonl"))


def test_a_pdf_with_no_text_is_refused_and_nothing_is_written(tmp_path) -> None:
    from rca.flows.ingest_flow import EmptyDocumentError
    from tests.pdf_fixtures import make_pdf

    path = tmp_path / "scanned.pdf"
    path.write_bytes(make_pdf([None, None]))
    flow = IngestFlow(settings=_settings_for(tmp_path))

    with pytest.raises(
        EmptyDocumentError, match=r"No text found in scanned\.pdf \(2 pages\).*scanned"
    ):
        flow.ingest_path(path)

    assert flow.graph_store.get_node("src:pdf/scanned") is None


def test_an_empty_note_is_refused(tmp_path) -> None:
    from rca.flows.ingest_flow import EmptyDocumentError

    path = tmp_path / "empty.md"
    path.write_text("   \n\n", encoding="utf-8")
    flow = IngestFlow(settings=_settings_for(tmp_path))

    with pytest.raises(EmptyDocumentError, match=r"No text found in empty\.md"):
        flow.ingest_path(path)
    assert flow.graph_store.get_node(make_source_id("note", "empty")) is None


def test_a_file_that_became_empty_keeps_its_previous_version(tmp_path) -> None:
    from rca.flows.ingest_flow import EmptyDocumentError

    path = tmp_path / "note.md"
    path.write_text("# Note\n\nreal content here\n", encoding="utf-8")
    flow = IngestFlow(settings=_settings_for(tmp_path))
    first = flow.ingest_path(path)

    path.write_text("", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        flow.ingest_path(path)

    assert flow.graph_store.list_source_chunk_ids(first.source_id) == first.chunk_ids
    assert len(flow.graph_store.list_source_revisions(first.source_id)) == 1


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("e.yaml", ""),
        ("t.yml", "~\n"),
        ("o.json", "{}"),
        ("zw.md", "​​\n"),
        ("nul.md", "\x00\x00"),
        ("hash.md", "#\n"),
    ],
)
def test_files_with_no_letters_or_digits_are_refused(tmp_path, name, body) -> None:
    from rca.flows.ingest_flow import EmptyDocumentError

    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    flow = IngestFlow(settings=_settings_for(tmp_path))

    with pytest.raises(EmptyDocumentError):
        flow.ingest_path(path)
