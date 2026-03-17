from __future__ import annotations

from pathlib import Path

from rca.config.settings import Settings
from rca.flows.ingest_flow import IngestFlow
from rca.flows.retrieve_flow import RetrieveFlow


def test_ingest_flow_indexes_markdown_note(tmp_path) -> None:
    runtime_dir = tmp_path / ".rca"
    settings = Settings(
        data_dir=runtime_dir,
        graph_db_path=runtime_dir / "graph.sqlite3",
        vector_dir=runtime_dir / "vectors",
        event_log_path=runtime_dir / "events.jsonl",
        telemetry_log_path=runtime_dir / "telemetry.jsonl",
        experiment_db_path=runtime_dir / "experiments.sqlite3",
        tool_policy_path=Path("rca/config/tool_policies.yaml"),
    )
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
    assert result.chunk_ids
    assert bundle.hits
    assert bundle.hits[0].metadata.get("source_id") == result.source_id


def test_ingest_flow_truncates_source_text_but_keeps_full_chunk_text(tmp_path) -> None:
    runtime_dir = tmp_path / ".rca"
    settings = Settings(
        data_dir=runtime_dir,
        graph_db_path=runtime_dir / "graph.sqlite3",
        vector_dir=runtime_dir / "vectors",
        event_log_path=runtime_dir / "events.jsonl",
        telemetry_log_path=runtime_dir / "telemetry.jsonl",
        experiment_db_path=runtime_dir / "experiments.sqlite3",
        tool_policy_path=Path("rca/config/tool_policies.yaml"),
    )
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
