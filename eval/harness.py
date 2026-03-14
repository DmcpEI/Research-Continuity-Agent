"""Generation evaluation harness for Research Continuity Agent."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from rca.flows.generate_flow import GenerateFlow


class GoldenPair(BaseModel):
    """Single golden evaluation pair."""

    id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)
    expected_source: str
    difficulty: Literal["easy", "medium", "hard"]

    model_config = {"extra": "ignore"}


class EvaluationCaseResult(BaseModel):
    """Per-question evaluation result."""

    id: str
    question: str
    difficulty: str
    expected_source: str
    expected_keywords: list[str] = Field(default_factory=list)
    answer: str
    grounded: bool
    citations: list[str] = Field(default_factory=list)
    source_correct: bool
    keyword_hits: float
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    latency_ms: float
    trace_path: str | None = None
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run generation evaluation against eval/golden.json.")
    parser.add_argument(
        "--golden-path",
        default=str(Path("eval") / "golden.json"),
        help="Path to the golden evaluation JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("eval") / "results"),
        help="Directory where run artifacts should be written.",
    )
    return parser


def load_golden_pairs(path: Path) -> list[GoldenPair]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_pairs = payload.get("pairs", [])
    elif isinstance(payload, list):
        raw_pairs = payload
    else:
        raise ValueError(f"Unsupported golden payload in {path}")
    return [GoldenPair.model_validate(item) for item in raw_pairs]


def evaluate_pair(flow: GenerateFlow, pair: GoldenPair) -> tuple[EvaluationCaseResult, dict[str, Any] | None]:
    started_at = perf_counter()
    try:
        generated = flow.generate_answer(pair.question)
        latency_ms = (perf_counter() - started_at) * 1000.0
        citation_source_ids = [citation.source_id for citation in generated.citations]
        matched_keywords, missing_keywords = keyword_matches(generated.answer, pair.expected_keywords)
        keyword_hits = calculate_keyword_hit_rate(matched_keywords, pair.expected_keywords)

        return (
            EvaluationCaseResult(
                id=pair.id,
                question=pair.question,
                difficulty=pair.difficulty,
                expected_source=pair.expected_source,
                expected_keywords=pair.expected_keywords,
                answer=generated.answer,
                grounded=generated.grounded,
                citations=citation_source_ids,
                source_correct=pair.expected_source in citation_source_ids,
                keyword_hits=keyword_hits,
                matched_keywords=matched_keywords,
                missing_keywords=missing_keywords,
                latency_ms=latency_ms,
            ),
            generated.trace.model_dump(mode="json") if generated.trace is not None else None,
        )
    except Exception as exc:
        latency_ms = (perf_counter() - started_at) * 1000.0
        return (
            EvaluationCaseResult(
                id=pair.id,
                question=pair.question,
                difficulty=pair.difficulty,
                expected_source=pair.expected_source,
                expected_keywords=pair.expected_keywords,
                answer="",
                grounded=False,
                citations=[],
                source_correct=False,
                keyword_hits=0.0,
                matched_keywords=[],
                missing_keywords=pair.expected_keywords,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )


def keyword_matches(answer: str, expected_keywords: list[str]) -> tuple[list[str], list[str]]:
    normalized_answer = answer.casefold()
    matched = [keyword for keyword in expected_keywords if keyword.casefold() in normalized_answer]
    missing = [keyword for keyword in expected_keywords if keyword not in matched]
    return matched, missing


def calculate_keyword_hit_rate(matched_keywords: list[str], expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    return len(matched_keywords) / len(expected_keywords)


def aggregate_results(results: list[EvaluationCaseResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "overall_grounded_rate": 0.0,
            "citation_precision": 0.0,
            "average_keyword_hit_rate": 0.0,
            "average_latency_ms": 0.0,
            "by_difficulty": {},
            "top_failures": [],
        }

    by_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [result for result in results if result.difficulty == difficulty]
        if not subset:
            continue
        by_difficulty[difficulty] = summarize_subset(subset)

    top_failures = sorted(results, key=lambda result: result.keyword_hits)[:5]
    return {
        "overall_grounded_rate": sum(result.grounded for result in results) / total,
        "citation_precision": sum(result.source_correct for result in results) / total,
        "average_keyword_hit_rate": sum(result.keyword_hits for result in results) / total,
        "average_latency_ms": sum(result.latency_ms for result in results) / total,
        "by_difficulty": by_difficulty,
        "top_failures": [
            {
                "id": result.id,
                "difficulty": result.difficulty,
                "keyword_hits": result.keyword_hits,
                "grounded": result.grounded,
                "source_correct": result.source_correct,
                "error": result.error,
            }
            for result in top_failures
        ],
    }


def summarize_subset(results: list[EvaluationCaseResult]) -> dict[str, Any]:
    total = len(results)
    return {
        "count": total,
        "grounded_rate": sum(result.grounded for result in results) / total,
        "citation_precision": sum(result.source_correct for result in results) / total,
        "average_keyword_hit_rate": sum(result.keyword_hits for result in results) / total,
        "average_latency_ms": sum(result.latency_ms for result in results) / total,
    }


def print_summary(summary: dict[str, Any], total_cases: int) -> None:
    print(f"Cases: {total_cases}")
    print(f"Overall grounded rate: {summary['overall_grounded_rate']:.1%}")
    print(f"Citation precision: {summary['citation_precision']:.1%}")
    print(f"Average keyword hit rate: {summary['average_keyword_hit_rate']:.3f}")
    print(f"Average latency: {summary['average_latency_ms']:.1f} ms")
    print("")
    print("Breakdown by difficulty:")
    for difficulty in ("easy", "medium", "hard"):
        metrics = summary["by_difficulty"].get(difficulty)
        if metrics is None:
            continue
        print(
            f"- {difficulty}: grounded={metrics['grounded_rate']:.1%}, "
            f"citation_precision={metrics['citation_precision']:.1%}, "
            f"keyword_hit_rate={metrics['average_keyword_hit_rate']:.3f}, "
            f"latency={metrics['average_latency_ms']:.1f} ms"
        )
    print("")
    print("Top 5 failures:")
    for failure in summary["top_failures"]:
        error_suffix = f", error={failure['error']}" if failure["error"] else ""
        print(
            f"- {failure['id']} ({failure['difficulty']}): "
            f"keyword_hits={failure['keyword_hits']:.3f}, "
            f"grounded={failure['grounded']}, "
            f"source_correct={failure['source_correct']}{error_suffix}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    golden_path = Path(args.golden_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    golden_pairs = load_golden_pairs(golden_path)
    flow = GenerateFlow()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_dir = output_dir / "traces" / timestamp
    trace_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvaluationCaseResult] = []
    for pair in golden_pairs:
        result, trace_payload = evaluate_pair(flow, pair)
        if trace_payload is not None:
            trace_path = trace_dir / f"{pair.id}.json"
            trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
            result.trace_path = str(trace_path)
        results.append(result)

    summary = aggregate_results(results)
    summary["trace_dir"] = str(trace_dir)

    output_path = output_dir / f"run_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "golden_path": str(golden_path),
        "cases": len(golden_pairs),
        "trace_dir": str(trace_dir),
        "summary": summary,
        "results": [result.model_dump(mode="json") for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_summary(summary, len(golden_pairs))
    print("")
    print(f"Saved full results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
