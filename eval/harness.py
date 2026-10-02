"""Generation evaluation harness for Research Continuity Agent."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from rca.config.settings import Settings
from rca.flows.generate_flow import GenerateFlow
from rca.llm.client import ChatMessage
from rca.llm.factory import get_llm_client


class GoldenPair(BaseModel):
    """Single golden evaluation pair."""

    id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)
    expected_source: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"]
    category: str | None = None
    answerable: bool = True
    negative_type: str | None = None
    notes: str | None = None

    model_config = {"extra": "ignore"}

    def expected_source_ids(self) -> list[str]:
        if self.expected_sources:
            return self.expected_sources
        return [self.expected_source] if self.expected_source else []


class EvaluationCaseResult(BaseModel):
    """Per-question evaluation result."""

    id: str
    question: str
    difficulty: str
    category: str | None = None
    answerable: bool = True
    negative_type: str | None = None
    expected_source: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    answer: str
    grounded: bool
    abstained: bool = False
    citations: list[str] = Field(default_factory=list)
    source_correct: bool
    keyword_hits: float
    max_retrieval_score: float = 0.0
    unique_sources_top5: int = 0
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    failure_labels: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    context_sources: list[str] = Field(default_factory=list)
    failure_stage: str = "ok"
    latency_ms: float
    trace_path: str | None = None
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run generation evaluation against eval/golden.json."
    )
    parser.add_argument(
        "--golden-path",
        default=str(Path("eval") / "golden.json"),
        help="Path to the golden evaluation JSON file.",
    )
    parser.add_argument(
        "--external-golden-path",
        default=None,
        help=(
            "Optional human-authored external golden file. When supplied, the harness "
            "runs this file instead of --golden-path and labels the dataset as external."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("eval") / "results"),
        help="Directory where run artifacts should be written.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Generation model for this eval run. Defaults to RCA_EVAL_MODEL, then "
            "RCA_GENERATION_MODEL / Settings().generation_model."
        ),
    )
    parser.add_argument(
        "--strict-model-check",
        action="store_true",
        help="Exit if the configured eval model is not loaded in Ollama.",
    )
    parser.add_argument(
        "--rewrite-model",
        default=None,
        help="Optional query-rewrite model override for this run.",
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


def evaluate_pair(
    flow: GenerateFlow, pair: GoldenPair
) -> tuple[EvaluationCaseResult, dict[str, Any] | None]:
    started_at = perf_counter()
    expected_sources = pair.expected_source_ids()
    try:
        generated = flow.generate_answer(pair.question)
        latency_ms = (perf_counter() - started_at) * 1000.0
        citation_source_ids = [citation.source_id for citation in generated.citations]
        matched_keywords, missing_keywords = keyword_matches(
            generated.answer, pair.expected_keywords
        )
        keyword_hits = calculate_keyword_hit_rate(matched_keywords, pair.expected_keywords)
        max_retrieval_score, unique_sources_top5 = trace_retrieval_features(generated.trace)
        retrieved_sources, context_sources = trace_sources(generated.trace)
        if not pair.answerable:
            source_correct = generated.abstained
        elif expected_sources:
            source_correct = (not generated.abstained) and all(
                source in citation_source_ids for source in expected_sources
            )
        else:
            source_correct = False

        return (
            EvaluationCaseResult(
                id=pair.id,
                question=pair.question,
                difficulty=pair.difficulty,
                category=pair.category,
                answerable=pair.answerable,
                negative_type=pair.negative_type,
                expected_source=pair.expected_source,
                expected_sources=expected_sources,
                expected_keywords=pair.expected_keywords,
                answer=generated.answer,
                grounded=generated.grounded,
                abstained=generated.abstained,
                citations=citation_source_ids,
                source_correct=source_correct,
                keyword_hits=keyword_hits,
                max_retrieval_score=max_retrieval_score,
                unique_sources_top5=unique_sources_top5,
                matched_keywords=matched_keywords,
                missing_keywords=missing_keywords,
                failure_labels=generated.failure_labels,
                retrieved_sources=retrieved_sources,
                context_sources=context_sources,
                failure_stage=classify_failure_stage(
                    pair,
                    abstained=generated.abstained,
                    source_correct=source_correct,
                    retrieved_sources=retrieved_sources,
                    context_sources=context_sources,
                ),
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
                category=pair.category,
                answerable=pair.answerable,
                negative_type=pair.negative_type,
                expected_source=pair.expected_source,
                expected_sources=expected_sources,
                expected_keywords=pair.expected_keywords,
                answer="",
                grounded=False,
                abstained=False,
                citations=[],
                source_correct=False,
                keyword_hits=0.0,
                max_retrieval_score=0.0,
                unique_sources_top5=0,
                matched_keywords=[],
                missing_keywords=pair.expected_keywords,
                failure_labels=["error"],
                failure_stage="error",
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


def trace_retrieval_features(trace: Any | None) -> tuple[float, int]:
    if trace is None:
        return 0.0, 0

    provenance = getattr(trace, "provenance", None) or []
    if not provenance:
        return 0.0, 0

    top_hits = provenance[:5]
    return (
        max(item.score for item in top_hits),
        len({GenerateFlow._resolve_source_id(item.node_id) for item in top_hits}),
    )


def trace_sources(trace: Any | None) -> tuple[list[str], list[str]]:
    """Ordered unique source IDs retrieved, and those passed to the LLM as context."""
    if trace is None:
        return [], []
    resolve = GenerateFlow._resolve_source_id
    retrieved = [resolve(item.node_id) for item in getattr(trace, "provenance", None) or []]
    context = [resolve(node_id) for node_id in getattr(trace, "context_node_ids", None) or []]
    return list(dict.fromkeys(retrieved)), list(dict.fromkeys(context))


def classify_failure_stage(
    pair: GoldenPair,
    *,
    abstained: bool,
    source_correct: bool,
    retrieved_sources: list[str],
    context_sources: list[str],
) -> str:
    """Attribute a case to the first pipeline stage that lost the expected evidence."""
    if not pair.answerable:
        return "ok" if abstained else "false_answer"
    if source_correct:
        return "ok"
    expected = pair.expected_source_ids()
    if not all(source in retrieved_sources for source in expected):
        return "retrieval_miss"
    if not all(source in context_sources for source in expected):
        return "context_miss"
    if abstained:
        return "false_abstention"
    return "citation_miss"


def count_failure_stages(results: list[EvaluationCaseResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.failure_stage] = counts.get(result.failure_stage, 0) + 1
    return dict(sorted(counts.items()))


def abstention_by_negative_type(results: list[EvaluationCaseResult]) -> dict[str, str]:
    groups: dict[str, list[bool]] = {}
    for result in results:
        if not result.answerable:
            groups.setdefault(result.negative_type or "untyped", []).append(result.abstained)
    return {key: f"{sum(values)}/{len(values)}" for key, values in sorted(groups.items())}


def aggregate_results(results: list[EvaluationCaseResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "overall_grounded_rate": 0.0,
            "citation_precision": 0.0,
            "citation_precision_cases": 0,
            "abstention_recall": 0.0,
            "abstention_cases": 0,
            "answerable_abstentions": 0,
            "average_keyword_hit_rate": 0.0,
            "average_latency_ms": 0.0,
            "failure_label_counts": {},
            "failure_stage_counts": {},
            "abstention_by_negative_type": {},
            "by_difficulty": {},
            "top_failures": [],
        }

    answerable = [result for result in results if result.answerable]
    negatives = [result for result in results if not result.answerable]
    scored_answerable = [result for result in answerable if not result.abstained]

    by_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [result for result in results if result.difficulty == difficulty]
        if not subset:
            continue
        by_difficulty[difficulty] = summarize_subset(subset)

    top_failures = sorted(results, key=lambda result: result.keyword_hits)[:5]
    return {
        "overall_grounded_rate": sum(result.grounded for result in results) / total,
        "citation_precision": (
            sum(result.source_correct for result in scored_answerable) / len(scored_answerable)
            if scored_answerable
            else 0.0
        ),
        "citation_precision_cases": len(scored_answerable),
        "abstention_recall": (
            sum(result.abstained for result in negatives) / len(negatives) if negatives else 0.0
        ),
        "abstention_cases": len(negatives),
        "answerable_abstentions": sum(result.abstained for result in answerable),
        "average_keyword_hit_rate": sum(result.keyword_hits for result in results) / total,
        "average_latency_ms": sum(result.latency_ms for result in results) / total,
        "failure_label_counts": count_failure_labels(results),
        "failure_stage_counts": count_failure_stages(results),
        "abstention_by_negative_type": abstention_by_negative_type(results),
        "by_difficulty": by_difficulty,
        "top_failures": [
            {
                "id": result.id,
                "difficulty": result.difficulty,
                "keyword_hits": result.keyword_hits,
                "grounded": result.grounded,
                "abstained": result.abstained,
                "source_correct": result.source_correct,
                "failure_labels": result.failure_labels,
                "error": result.error,
            }
            for result in top_failures
        ],
    }


def count_failure_labels(results: list[EvaluationCaseResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for label in result.failure_labels:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def summarize_subset(results: list[EvaluationCaseResult]) -> dict[str, Any]:
    total = len(results)
    answerable = [result for result in results if result.answerable]
    negatives = [result for result in results if not result.answerable]
    scored_answerable = [result for result in answerable if not result.abstained]
    return {
        "count": total,
        "grounded_rate": sum(result.grounded for result in results) / total,
        "citation_precision": (
            sum(result.source_correct for result in scored_answerable) / len(scored_answerable)
            if scored_answerable
            else 0.0
        ),
        "abstention_recall": (
            sum(result.abstained for result in negatives) / len(negatives) if negatives else 0.0
        ),
        "answerable_abstentions": sum(result.abstained for result in answerable),
        "average_keyword_hit_rate": sum(result.keyword_hits for result in results) / total,
        "average_latency_ms": sum(result.latency_ms for result in results) / total,
    }


def print_summary(summary: dict[str, Any], total_cases: int) -> None:
    print(f"Cases: {total_cases}")
    print(f"Overall grounded rate: {summary['overall_grounded_rate']:.1%}")
    print(
        "Citation precision (answerable, non-abstained only): "
        f"{summary['citation_precision']:.1%} "
        f"over {summary['citation_precision_cases']} cases"
    )
    print(
        "Abstention recall (negative questions): "
        f"{summary['abstention_recall']:.1%} "
        f"over {summary['abstention_cases']} cases"
    )
    print(f"Answerable abstentions: {summary['answerable_abstentions']}")
    print(f"Average keyword hit rate: {summary['average_keyword_hit_rate']:.3f}")
    print(f"Average latency: {summary['average_latency_ms']:.1f} ms")
    if summary.get("failure_label_counts"):
        labels = ", ".join(
            f"{label}={count}" for label, count in summary["failure_label_counts"].items()
        )
        print(f"Failure labels: {labels}")
    if summary.get("failure_stage_counts"):
        stages = ", ".join(
            f"{stage}={count}" for stage, count in summary["failure_stage_counts"].items()
        )
        print(f"Failure stages: {stages}")
    if summary.get("abstention_by_negative_type"):
        by_type = ", ".join(
            f"{kind}={rate}" for kind, rate in summary["abstention_by_negative_type"].items()
        )
        print(f"Abstention by negative type: {by_type}")
    print("")
    print("Breakdown by difficulty:")
    for difficulty in ("easy", "medium", "hard"):
        metrics = summary["by_difficulty"].get(difficulty)
        if metrics is None:
            continue
        print(
            f"- {difficulty}: grounded={metrics['grounded_rate']:.1%}, "
            f"citation_precision={metrics['citation_precision']:.1%}, "
            f"abstention_recall={metrics['abstention_recall']:.1%}, "
            f"answerable_abstentions={metrics['answerable_abstentions']}, "
            f"keyword_hit_rate={metrics['average_keyword_hit_rate']:.3f}, "
            f"latency={metrics['average_latency_ms']:.1f} ms"
        )
    print("")
    print("Top 5 failures:")
    for failure in summary["top_failures"]:
        error_suffix = f", error={failure['error']}" if failure["error"] else ""
        labels_suffix = (
            f", labels={','.join(failure['failure_labels'])}" if failure["failure_labels"] else ""
        )
        print(
            f"- {failure['id']} ({failure['difficulty']}): "
            f"keyword_hits={failure['keyword_hits']:.3f}, "
            f"grounded={failure['grounded']}, "
            f"abstained={failure['abstained']}, "
            f"source_correct={failure['source_correct']}{labels_suffix}{error_suffix}"
        )


def validate_eval_llm(flow: GenerateFlow) -> None:
    try:
        flow.llm.chat([ChatMessage(role="user", content="ping")])
    except Exception as exc:
        model = getattr(flow.llm, "model", "unknown")
        base_url = getattr(flow.llm, "base_url", "unknown")
        raise RuntimeError(
            "LLM validation failed before eval run. "
            f"Model='{model}', base_url='{base_url}'. "
            "Check RCA_GENERATION_MODEL or available local models. "
            f"Error: {type(exc).__name__}: {exc}"
        ) from exc


def fetch_available_ollama_models(base_url: str = "http://localhost:11434") -> set[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError):
        return set()

    models = payload.get("models", []) if isinstance(payload, dict) else []
    loaded: set[str] = set()
    for model in models:
        name = model.get("name")
        if name:
            loaded.add(name)
    return loaded


def build_settings(eval_model: str | None) -> Settings:
    model = eval_model or os.environ.get("RCA_EVAL_MODEL")
    base_settings = Settings()
    if model:
        if base_settings.llm_backend == "openai_compatible":
            return Settings(openai_chat_model=model)
        return Settings(generation_model=model)
    return base_settings


def build_run_config(
    settings: Settings,
    flow: GenerateFlow,
    dataset_kind: str,
) -> dict[str, Any]:
    return {
        "dataset_kind": dataset_kind,
        "llm_backend": settings.llm_backend,
        "generation_model": settings.generation_model
        if settings.llm_backend == "ollama"
        else settings.openai_chat_model,
        "resolved_model": getattr(flow.llm, "model", flow.llm.__class__.__name__),
        "resolved_rewrite_model": getattr(
            flow.rewrite_llm, "model", flow.rewrite_llm.__class__.__name__
        ),
        "embedding_model": settings.embedding_model
        if settings.llm_backend == "ollama"
        else settings.openai_embed_model,
        "graph_db_path": str(settings.graph_db_path),
        "vector_dir": str(settings.vector_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    golden_path = Path(args.external_golden_path or args.golden_path)
    dataset_kind = "external" if args.external_golden_path else "golden"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    golden_pairs = load_golden_pairs(golden_path)
    settings = build_settings(args.model)
    loaded_models = (
        fetch_available_ollama_models(settings.llm_base_url)
        if settings.llm_backend == "ollama"
        else set()
    )
    if loaded_models and settings.generation_model not in loaded_models:
        warning = (
            f"Warning: configured model {settings.generation_model} is not available in Ollama. "
            f"Load it with: ollama pull {settings.generation_model}"
        )
        print(warning)
        if args.strict_model_check:
            return 3
    flow = GenerateFlow(settings=settings)
    if args.rewrite_model and args.rewrite_model.strip():
        rewrite_model = args.rewrite_model.strip()
        rewrite_settings = settings.model_copy(deep=True)
        rewrite_settings.generation_model = rewrite_model
        rewrite_settings.openai_chat_model = rewrite_model
        flow.rewrite_llm = get_llm_client(rewrite_settings)
    try:
        validate_eval_llm(flow)
    except RuntimeError as exc:
        print(str(exc))
        return 2
    run_config = build_run_config(settings, flow, dataset_kind)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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
        "dataset_kind": dataset_kind,
        "golden_path": str(golden_path),
        "cases": len(golden_pairs),
        "run_config": run_config,
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
