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
    note_path.write_text("# Demo Note\n\nThis repo tracks research continuity across experiments.\n", encoding="utf-8")

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
