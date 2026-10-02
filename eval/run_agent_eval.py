"""Agent benchmark: tool selection, steps, and task success for the multi-turn agent loop.

Each run sandboxes the agent: filesystem tools see a temporary copy of
eval/agent_fixture/, experiment tools see a freshly seeded experiments DB, and
knowledge-base search uses the live corpus read-only. Writes
eval/results/agent_eval_<timestamp>.json with per-task traces and provenance,
even if the run is interrupted ("complete": false). Fixture files and SEED_RUNS
must stay in sync with eval/agent_tasks.json; tests check the key values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca.agent.loop import AgentLoop  # noqa: E402
from rca.config.settings import Settings, get_settings  # noqa: E402
from rca.flows.retrieve_flow import RetrieveFlow  # noqa: E402
from rca.mcp_servers.experiments.server import ExperimentServer  # noqa: E402
from rca.store.graph_store import GraphStore  # noqa: E402
from rca.store.vector_store import VectorStore  # noqa: E402
from rca.telemetry.provenance import collect_provenance, finish_provenance  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "eval" / "agent_fixture"

# Runs the experiment-tool tasks in agent_tasks.json are written against.
SEED_RUNS = [
    {
        "run_id": "exp-pipeline-a",
        "name": "grocery-bagging-pipeline-a",
        "status": "complete",
        "metrics": {"accuracy": 0.81, "latency_s": 2.4},
        "metadata": {"config": "vlm_only"},
    },
    {
        "run_id": "exp-pipeline-b",
        "name": "grocery-bagging-pipeline-b",
        "status": "complete",
        "metrics": {"accuracy": 0.89, "latency_s": 3.1},
        "metadata": {"config": "yolo_vlm"},
    },
    {
        "run_id": "exp-packing-order",
        "name": "packing-order-batch16",
        "status": "failed",
        "metadata": {"error": "CUDA out of memory"},
    },
    {"run_id": "exp-scene-graph", "name": "scene-graph-planning", "status": "running"},
]


MALFORMED_CALL_WARNINGS = ("malformed", "without a name")


def is_invalid_call(call: dict[str, Any]) -> bool:
    """A tool call that did not do useful work: transport/handler error or an error string.

    MCP servers report many failures (path escapes, missing files) as successful calls
    whose text starts with "Error:", so status alone undercounts.
    """
    return call.get("status") != "success" or str(call.get("output", "")).startswith("Error:")


def score_task(
    task: dict[str, Any],
    answer: str,
    tools_called: list[str],
    iterations: int,
    error: str | None = None,
    stopped_reason: str = "final_answer",
) -> dict[str, Any]:
    """Score one agent run against a task spec (pure; no LLM)."""
    text = answer.casefold()
    required = task.get("required_tools", [])
    forbidden = task.get("forbidden_tools", [])
    missing_tools = [tool for tool in required if tool not in tools_called]
    forbidden_used = sorted({tool for tool in forbidden if tool in tools_called})
    missing_all = [kw for kw in task.get("answer_all", []) if kw.casefold() not in text]
    answer_any = task.get("answer_any", [])
    any_ok = not answer_any or any(kw.casefold() in text for kw in answer_any)
    leaked = [kw for kw in task.get("answer_none", []) if kw.casefold() in text]
    within_steps = iterations <= task.get("max_steps", 10)
    # The loop's automatic KB-search fallback is not the model choosing a tool.
    model_finished = stopped_reason == "final_answer"

    tool_selection_ok = not missing_tools and not forbidden_used and model_finished
    answer_ok = not missing_all and any_ok and not leaked
    needed = set(required)
    precision = (
        sum(tool in needed for tool in tools_called) / len(tools_called)
        if needed and tools_called
        else None
    )
    reasons = []
    if error:
        reasons.append(f"error: {error}")
    if not model_finished:
        reasons.append(f"stopped: {stopped_reason}")
    if missing_tools:
        reasons.append(f"missing tools: {missing_tools}")
    if forbidden_used:
        reasons.append(f"forbidden tools used: {forbidden_used}")
    if missing_all:
        reasons.append(f"answer missing: {missing_all}")
    if not any_ok:
        reasons.append(f"answer has none of: {answer_any}")
    if leaked:
        reasons.append(f"answer contains forbidden text: {leaked}")
    if not within_steps:
        reasons.append(f"{iterations} steps > max {task.get('max_steps', 10)}")
    return {
        "success": not error and tool_selection_ok and answer_ok and within_steps,
        "tool_selection_ok": tool_selection_ok,
        "answer_ok": answer_ok,
        "within_steps": within_steps,
        "tool_precision": precision,
        "unrequested_tool_calls": sum(tool not in needed for tool in tools_called),
        "reasons": reasons,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(values: list[bool]) -> float | None:
        return mean(values) if values else None

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        # Tool selection is only meaningful where the task requires tools.
        with_tools = [r for r in subset if r.get("required_tools")]
        precisions = [r["score"]["tool_precision"] for r in subset]
        precisions = [p for p in precisions if p is not None]
        stops: dict[str, int] = defaultdict(int)
        for r in subset:
            stops[r["stopped_reason"] or "unknown"] += 1
        return {
            "tasks": len(subset),
            "success_rate": rate([r["score"]["success"] for r in subset]),
            "tool_selection_accuracy": rate([r["score"]["tool_selection_ok"] for r in with_tools]),
            "tool_selection_tasks": len(with_tools),
            "mean_tool_precision": mean(precisions) if precisions else None,
            "answer_accuracy": rate([r["score"]["answer_ok"] for r in subset]),
            "mean_steps": mean(r["iterations"] for r in subset),
            "mean_tool_calls": mean(len(r["tool_calls"]) for r in subset),
            "unrequested_tool_calls": sum(r["score"]["unrequested_tool_calls"] for r in subset),
            "invalid_tool_calls": sum(
                sum(is_invalid_call(call) for call in r["tool_calls"]) for r in subset
            ),
            "malformed_tool_calls": sum(
                sum(any(m in w for m in MALFORMED_CALL_WARNINGS) for w in r["warnings"])
                for r in subset
            ),
            "stopped_reasons": dict(sorted(stops.items())),
            "mean_latency_s": mean(r["latency_ms"] for r in subset) / 1000.0,
        }

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    return {
        "overall": block(rows),
        "by_category": {k: block(v) for k, v in sorted(by_category.items())},
    }


def benchmark_fingerprint(tasks_path: Path) -> str:
    """Hash of the task file, fixture tree, and seeded runs; untracked files escape git_state."""
    digest = hashlib.sha256(tasks_path.read_bytes())
    for path in sorted(p for p in FIXTURE_DIR.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(FIXTURE_DIR)).encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    digest.update(json.dumps(SEED_RUNS, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]


def seed_experiments(db_path: Path) -> None:
    server = ExperimentServer(db_path)
    for run in SEED_RUNS:
        server.record_run(**run)


def sandbox_settings(base: Settings, sandbox: Path, model: str | None) -> Settings:
    """Settings that point the agent's MCP servers at the sandbox, never at real data."""
    return base.model_copy(
        update={
            "filesystem_root": sandbox / "files",
            "experiment_db_path": sandbox / "experiments.sqlite3",
            "enable_filesystem_tools": True,
            **({"agent_model": model} if model else {}),
        }
    )


def run_task(loop: AgentLoop, task: dict[str, Any]) -> dict[str, Any]:
    try:
        result = loop.run(task["query"])
    except Exception as exc:  # one broken task must not lose the rest of a long run
        error = f"{type(exc).__name__}: {exc}"
        return {
            "id": task["id"],
            "category": task["category"],
            "query": task["query"],
            "required_tools": task.get("required_tools", []),
            "answer": "",
            "error": error,
            "iterations": 0,
            "tool_calls": [],
            "warnings": [],
            "stopped_reason": "exception",
            "latency_ms": 0.0,
            "model": "",
            "score": score_task(task, "", [], 0, error, "exception"),
        }
    calls = [call.model_dump(mode="json") for call in result.trace.tool_calls]
    tools_called = [call["tool_name"] for call in calls]
    return {
        "id": task["id"],
        "category": task["category"],
        "query": task["query"],
        "required_tools": task.get("required_tools", []),
        "answer": result.answer,
        "error": result.error,
        "iterations": result.trace.iterations,
        "tool_calls": calls,
        "warnings": result.trace.warnings,
        "stopped_reason": result.trace.stopped_reason,
        "latency_ms": result.trace.total_latency_ms,
        "model": result.trace.model,
        "score": score_task(
            task,
            result.answer,
            tools_called,
            result.trace.iterations,
            result.error,
            result.trace.stopped_reason,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(REPO_ROOT / "eval" / "agent_tasks.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "eval" / "results"))
    parser.add_argument("--model", default=None, help="Agent model (default: RCA_AGENT_MODEL).")
    parser.add_argument("--only", default=None, help="Comma-separated task IDs to run.")
    args = parser.parse_args(argv)

    tasks_path = Path(args.tasks)
    spec = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {task["id"] for task in tasks}
        if unknown:
            print(f"Unknown task IDs: {sorted(unknown)}")
            return 2
        tasks = [task for task in tasks if task["id"] in wanted]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir) / f"agent_eval_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    settings: Settings | None = None
    complete = False
    sandbox = Path(tempfile.mkdtemp(prefix="rca-agent-eval-"))
    try:
        shutil.copytree(FIXTURE_DIR, sandbox / "files")
        seed_experiments(sandbox / "experiments.sqlite3")
        settings = sandbox_settings(get_settings(), sandbox, args.model)
        graph_store = GraphStore(settings.graph_db_path)
        vector_store = VectorStore(settings.vector_dir, settings.default_collection)
        retrieve = RetrieveFlow(
            settings=settings, graph_store=graph_store, vector_store=vector_store
        )
        provenance = collect_provenance(settings, graph_store, vector_store)
        provenance["benchmark_sha256"] = benchmark_fingerprint(tasks_path)
        loop = AgentLoop(settings=settings, retrieve_flow=retrieve)
        try:
            for index, task in enumerate(tasks, 1):
                row = run_task(loop, task)
                rows.append(row)
                score = row["score"]
                status = "ok" if score["success"] else "FAIL " + "; ".join(score["reasons"])
                called = [call["tool_name"] for call in row["tool_calls"]]
                print(f"[{index}/{len(tasks)}] {task['id']}: {called} -> {status}", flush=True)
        finally:
            loop.close()
        provenance = finish_provenance(provenance, vector_store)
        complete = True
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
        # Always persist what ran, even on Ctrl-C or a startup failure mid-run.
        summary = summarize(rows) if rows else {}
        output.write_text(
            json.dumps(
                {
                    "complete": complete,
                    "tasks_path": str(tasks_path),
                    "tasks_run": len(rows),
                    "agent_model": settings.agent_model if settings else args.model,
                    "summary": summary,
                    "provenance": provenance,
                    "results": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if summary:
            overall = summary["overall"]
            selection = overall["tool_selection_accuracy"]
            print(
                f"success {overall['success_rate']:.0%} | tool selection "
                f"{'n/a' if selection is None else f'{selection:.0%}'} | mean steps "
                f"{overall['mean_steps']:.1f} | invalid tool calls {overall['invalid_tool_calls']}"
            )
        print(f"Wrote {output}{'' if complete else ' (partial)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
