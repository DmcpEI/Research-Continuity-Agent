from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_agent_eval", ROOT / "eval" / "run_agent_eval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_score_task_requires_tools_answer_and_step_budget() -> None:
    agent_eval = _load()
    task = {
        "required_tools": ["read_text_file"],
        "forbidden_tools": ["search_text"],
        "answer_all": ["16"],
        "answer_any": ["yolov8n", "YOLO"],
        "answer_none": ["root:x:"],
        "max_steps": 3,
    }

    ok = agent_eval.score_task(task, "Batch size 16, detector YOLOv8n.", ["read_text_file"], 2)
    assert ok["success"] and ok["tool_selection_ok"] and ok["answer_ok"] and ok["reasons"] == []

    cases = {
        "missing tools": agent_eval.score_task(task, "16 yolov8n", ["list_directory"], 2),
        "forbidden tools": agent_eval.score_task(
            task, "16 yolov8n", ["read_text_file", "search_text"], 2
        ),
        "answer missing": agent_eval.score_task(task, "yolov8n", ["read_text_file"], 2),
        "answer has none of": agent_eval.score_task(task, "16 only", ["read_text_file"], 2),
        "forbidden text": agent_eval.score_task(task, "16 yolov8n root:x:0", ["read_text_file"], 2),
        "steps": agent_eval.score_task(task, "16 yolov8n", ["read_text_file"], 4),
        "error": agent_eval.score_task(task, "16 yolov8n", ["read_text_file"], 2, error="boom"),
    }
    for reason, result in cases.items():
        assert result["success"] is False, reason
        assert any(reason in text for text in result["reasons"]), (reason, result["reasons"])


def test_task_file_is_consistent_with_fixture_and_tools() -> None:
    spec = json.loads((ROOT / "eval" / "agent_tasks.json").read_text(encoding="utf-8"))
    known_tools = {
        "search_knowledge_base",
        "list_directory",
        "read_text_file",
        "search_text",
        "list_runs",
        "get_run",
    }
    ids = [task["id"] for task in spec["tasks"]]
    assert len(ids) == len(set(ids)) == 20
    for task in spec["tasks"]:
        assert set(task.get("required_tools", [])) <= known_tools, task["id"]
        assert task["category"] in {
            "knowledge_base",
            "filesystem",
            "experiments",
            "multi_step",
            "refusal",
        }
    fixture = (ROOT / "eval" / "agent_fixture" / "configs" / "pipeline.yaml").read_text()
    assert "batch_size: 16" in fixture and "yolov8n" in fixture


def test_score_task_rejects_fallback_answers_and_reports_precision() -> None:
    agent_eval = _load()
    task = {"required_tools": ["search_knowledge_base"], "answer_any": ["cuboid"], "max_steps": 4}

    fallback = agent_eval.score_task(
        task, "cuboid objects", ["search_knowledge_base"], 1, stopped_reason="fallback_no_tools"
    )
    assert fallback["success"] is False and fallback["tool_selection_ok"] is False
    assert any("stopped: fallback_no_tools" in reason for reason in fallback["reasons"])

    wandering = agent_eval.score_task(
        task, "cuboid", ["search_knowledge_base", "read_text_file", "list_directory"], 4
    )
    assert wandering["success"] is True
    assert wandering["tool_precision"] == 1 / 3
    assert wandering["unrequested_tool_calls"] == 2
    assert agent_eval.score_task({}, "x", [], 1)["tool_precision"] is None


def test_summarize_counts_error_text_calls_and_skips_tool_free_tasks() -> None:
    agent_eval = _load()

    def row(task, calls, stopped="final_answer", warnings=()):
        names = [call["tool_name"] for call in calls]
        return {
            "id": task["id"],
            "category": "c",
            "required_tools": task.get("required_tools", []),
            "iterations": 2,
            "tool_calls": calls,
            "warnings": list(warnings),
            "stopped_reason": stopped,
            "latency_ms": 1000.0,
            "score": agent_eval.score_task(task, "ok", names, 2, stopped_reason=stopped),
        }

    ok_call = {"tool_name": "read_text_file", "status": "success", "output": "batch_size: 16"}
    err_text = {
        "tool_name": "read_text_file",
        "status": "success",
        "output": "Error: Path escapes root",
    }
    failed = {"tool_name": "nope", "status": "error", "output": "Unknown tool: nope"}
    rows = [
        row({"id": "a", "required_tools": ["read_text_file"]}, [ok_call]),
        row({"id": "b", "required_tools": ["read_text_file"]}, [err_text, failed]),
        row({"id": "c"}, [], warnings=["tool call arguments for x were malformed"]),
    ]
    overall = agent_eval.summarize(rows)["overall"]
    assert overall["invalid_tool_calls"] == 2
    assert overall["malformed_tool_calls"] == 1
    assert overall["tool_selection_tasks"] == 2  # tool-free task "c" is excluded
    assert overall["tool_selection_accuracy"] == 1.0
    assert overall["stopped_reasons"] == {"final_answer": 3}


def test_sandbox_settings_point_mcp_servers_at_the_sandbox(tmp_path) -> None:
    from rca.agent.mcp import MCPClientManager
    from rca.config.settings import Settings

    agent_eval = _load()
    settings = agent_eval.sandbox_settings(Settings(), tmp_path, "qwen3.5:9b")
    params = MCPClientManager(settings)._server_parameters()

    assert settings.agent_model == "qwen3.5:9b"
    assert str((tmp_path / "files").resolve()) in params["filesystem"].args
    assert str((tmp_path / "experiments.sqlite3").resolve()) in params["experiments"].args


def test_fixture_and_seed_data_match_task_expectations() -> None:
    agent_eval = _load()
    fixture = ROOT / "eval" / "agent_fixture"
    meeting = (fixture / "notes" / "meeting-2026-03-04.md").read_text(encoding="utf-8")
    reading = (fixture / "notes" / "reading-list.md").read_text(encoding="utf-8")
    seeds = {run["run_id"]: run for run in agent_eval.SEED_RUNS}

    assert "2026-03-20" in meeting and "Pipeline B" in meeting and "SayPlan" in meeting
    assert "JSONSchemaBench" in reading and "(to read)" in reading
    assert seeds["exp-pipeline-b"]["metrics"]["accuracy"] == 0.89
    assert seeds["exp-pipeline-b"]["metadata"]["config"] == "yolo_vlm"
    assert "CUDA" in seeds["exp-packing-order"]["metadata"]["error"]
    assert seeds["exp-scene-graph"]["status"] == "running"
    assert len(agent_eval.benchmark_fingerprint(ROOT / "eval" / "agent_tasks.json")) == 16


def test_script_entry_point_runs_main() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_agent_eval.py"), "--only", "no-such-task"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2, result.stderr
    assert "Unknown task IDs: ['no-such-task']" in result.stdout
