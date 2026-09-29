from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rca.api.main import create_app
from rca.config.settings import Settings
from rca.contracts.nodes import Node, NodeKind
from rca.store.graph_store import GraphStore


def _settings_for(tmp_path: Path) -> Settings:
    runtime_dir = tmp_path / ".rca"
    return Settings(
        workspace_root=tmp_path,
        filesystem_root=tmp_path,
        data_dir=runtime_dir,
        graph_db_path=runtime_dir / "graph.sqlite3",
        vector_dir=runtime_dir / "vectors",
        event_log_path=runtime_dir / "events.jsonl",
        telemetry_log_path=runtime_dir / "telemetry.jsonl",
        experiment_db_path=runtime_dir / "experiments.sqlite3",
        tool_policy_path=Path("rca/config/tool_policies.yaml"),
    )


def test_pdf_endpoint_serves_pdf_for_source(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%minimal\n")

    source_id = "src:pdf/sample-paper"
    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="Sample Paper",
            metadata={"latest_path": str(pdf_path)},
        )
    )

    client = TestClient(create_app(settings=settings, graph_store=store))
    response = client.get(f"/sources/{source_id}/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-disposition"].startswith("inline")


def test_pdf_endpoint_resolves_chunk_to_parent_source(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%minimal\n")

    source_id = "src:pdf/sample-paper"
    chunk_id = "chk:pdf/sample-paper:0000"
    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="Sample Paper",
            metadata={"latest_path": str(pdf_path)},
        )
    )
    store.upsert_node(
        Node(
            id=chunk_id,
            kind=NodeKind.chunk,
            title="Chunk",
            metadata={"source_id": source_id},
        )
    )

    client = TestClient(create_app(settings=settings, graph_store=store))
    response = client.get(f"/sources/{chunk_id}/pdf")

    assert response.status_code == 200


def test_pdf_endpoint_returns_404_for_missing_source(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)

    client = TestClient(create_app(settings=settings, graph_store=store))
    response = client.get("/sources/src:pdf/missing/pdf")

    assert response.status_code == 404


def test_pdf_endpoint_blocks_paths_outside_root(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    outside_path = Path("/tmp/outside.pdf")

    source_id = "src:pdf/outside"
    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="Outside",
            metadata={"latest_path": str(outside_path)},
        )
    )

    client = TestClient(create_app(settings=settings, graph_store=store))
    response = client.get(f"/sources/{source_id}/pdf")

    assert response.status_code == 403


def test_pdf_endpoint_returns_404_for_non_pdf_path(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    store = GraphStore(settings.graph_db_path)
    txt_path = tmp_path / "paper.txt"
    txt_path.write_text("not a pdf", encoding="utf-8")

    source_id = "src:pdf/not-pdf"
    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="Not PDF",
            metadata={"latest_path": str(txt_path)},
        )
    )

    client = TestClient(create_app(settings=settings, graph_store=store))
    response = client.get(f"/sources/{source_id}/pdf")

    assert response.status_code == 404
