"""Analyze low keyword-hit generation cases to guide rewrite prompt tuning.

Usage:
  uv run python eval/analyze_keyword_failures.py
  uv run python eval/analyze_keyword_failures.py --run-file eval/results/run_20260421T111146Z.json --top-n 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rca.flows.retrieve_flow import RetrieveFlow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect lowest keyword-hit cases from a harness run and analyze retrieval mismatch."
    )
    parser.add_argument(
        "--run-file",
        default=None,
        help="Path to a harness run artifact (defaults to latest eval/results/run_*.json).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of lowest keyword-hit cases to inspect.",
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=5,
        help="Top-k retrieval hits to print per case.",
    )
    return parser


def find_latest_run(results_dir: Path) -> Path:
    candidates = sorted(
        results_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No run_*.json files found under {results_dir}")
    return candidates[0]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_sort_key(item: dict[str, Any]) -> tuple[float, float]:
    return (float(item.get("keyword_hits", 0.0)), float(item.get("latency_ms", 0.0)))


def load_trace(trace_path: str | None) -> dict[str, Any] | None:
    if not trace_path:
        return None
    path = Path(trace_path)
    if not path.exists():
        return None
    return load_json(path)


def normalize(text: str) -> str:
    return text.casefold()


def analyze_case(
    retrieve_flow: RetrieveFlow,
    case: dict[str, Any],
    retrieval_limit: int,
) -> dict[str, Any]:
    trace_payload = load_trace(case.get("trace_path"))
    rewritten_query = None
    if trace_payload is not None:
        rewritten_query = trace_payload.get("rewritten_query")

    search_query = rewritten_query or case.get("question", "")
    bundle = retrieve_flow.retrieve(search_query, limit=retrieval_limit)

    expected_keywords = case.get("expected_keywords", [])
    missing_keywords = case.get("missing_keywords", expected_keywords)

    retrieval_corpus = "\n".join([f"{hit.title}\n{hit.excerpt}" for hit in bundle.hits])
    retrieval_norm = normalize(retrieval_corpus)

    missing_in_retrieval = [
        keyword for keyword in missing_keywords if normalize(keyword) not in retrieval_norm
    ]
    present_in_retrieval = [
        keyword for keyword in missing_keywords if normalize(keyword) in retrieval_norm
    ]

    return {
        "rewritten_query": rewritten_query,
        "search_query": search_query,
        "hits": [
            {
                "node_id": hit.node_id,
                "score": hit.score,
                "title": hit.title,
                "excerpt": hit.excerpt,
            }
            for hit in bundle.hits
        ],
        "missing_in_retrieval": missing_in_retrieval,
        "present_in_retrieval": present_in_retrieval,
    }


def print_case(case: dict[str, Any], analysis: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(
        f"Case: {case.get('id')} | difficulty={case.get('difficulty')} | answerable={case.get('answerable')}"
    )
    print(f"Keyword hit: {case.get('keyword_hits', 0.0):.3f}")
    print(f"Question: {case.get('question')}")
    print(f"Expected keywords: {case.get('expected_keywords', [])}")
    print(f"Missing keywords: {case.get('missing_keywords', [])}")
    print(f"Rewritten query: {analysis.get('rewritten_query')}")
    print(f"Effective query: {analysis.get('search_query')}")

    print("\nMissing-keyword diagnosis:")
    print(f"- missing in retrieval context: {analysis.get('missing_in_retrieval', [])}")
    print(
        f"- present in retrieval but absent in answer: {analysis.get('present_in_retrieval', [])}"
    )

    print("\nTop retrieval hits:")
    for index, hit in enumerate(analysis.get("hits", []), start=1):
        excerpt = str(hit.get("excerpt", "")).replace("\n", " ").strip()
        print(
            f"{index}. {hit.get('node_id')} | score={float(hit.get('score', 0.0)):.3f} | "
            f"title={hit.get('title')}"
        )
        print(f"   excerpt: {excerpt[:260]}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_file = Path(args.run_file) if args.run_file else find_latest_run(Path("eval") / "results")
    payload = load_json(run_file)

    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("Run artifact does not contain a results array.")

    # Focus on answerable cases where keyword coverage matters.
    answerable_cases = [case for case in raw_results if case.get("answerable", True)]
    worst_cases = sorted(answerable_cases, key=case_sort_key)[: max(1, args.top_n)]

    print(f"Run file: {run_file}")
    print(f"Total answerable cases: {len(answerable_cases)}")
    print(f"Inspecting worst {len(worst_cases)} cases by keyword hit rate")

    retrieve_flow = RetrieveFlow()

    for case in worst_cases:
        analysis = analyze_case(retrieve_flow, case, args.retrieval_limit)
        print_case(case, analysis)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
