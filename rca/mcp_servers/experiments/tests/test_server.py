from __future__ import annotations

from rca.mcp_servers.experiments.server import ExperimentServer


def test_experiment_server_records_and_lists_runs(tmp_path) -> None:
    server = ExperimentServer(tmp_path / "experiments.sqlite3")

    created = server.record_run("baseline", status="completed", metrics={"accuracy": 0.92})
    runs = server.list_runs()

    assert created.name == "baseline"
    assert runs
    assert runs[0].run_id == created.run_id
