"""Retrieval regression gate: fail CI when a change pushes fixture questions down the ranking.

Runs GenerateFlow's retrieval call (limit=5, classified query type) on a synthetic corpus
without Ollama: the vector store is forced onto its deterministic JSON backend, and the
reranker and query rewrite are off. That covers FTS5 candidates, lexical scoring, and
tokenisation. Not covered: the vector component (the JSON fallback is weak, so breaking it
does not lower ranks here), most query-type weights, and source expansion and the reserve
slot, which cannot change the top-5 sources without the reranker. See docs/EVAL.md.
After a deliberate retrieval change, refresh the baseline with
RCA_UPDATE_RETRIEVAL_BASELINE=1 uv run pytest tests/integration/test_retrieval_gate.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rca.config.settings import Settings
from rca.flows.ingest_flow import IngestFlow
from rca.flows.retrieve_flow import RetrieveFlow, _source_of
from rca.retrieval.query_classifier import classify_query
from rca.store import vector_store

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "retrieval_gate"
BASELINE_PATH = FIXTURE_DIR / "baseline.json"
K = 5


def _settings(tmp_path: Path) -> Settings:
    runtime = tmp_path / ".rca"
    # Pin every retrieval-affecting field: Settings also reads the environment and .env.
    return Settings(
        data_dir=runtime,
        graph_db_path=runtime / "graph.sqlite3",
        vector_dir=runtime / "vectors",
        event_log_path=runtime / "events.jsonl",
        telemetry_log_path=runtime / "telemetry.jsonl",
        experiment_db_path=runtime / "experiments.sqlite3",
        tool_policy_path=Path("rca/config/tool_policies.yaml"),
        default_collection="retrieval-gate",
        chunk_size=400,
        chunk_overlap=50,
        enable_reranker=False,
        retrieval_reserve_new_source=True,
        retrieval_reserve_min_rerank_score=-6.0,
    )


def _ranks(tmp_path: Path) -> dict[str, int | None]:
    """Per question: hit position (1-based) of the last expected source in the top K, None if any is missing."""
    settings = _settings(tmp_path)
    ingest = IngestFlow(settings=settings)
    assert ingest.vector_store.backend == "json"
    for path in sorted((FIXTURE_DIR / "corpus").glob("*.md")):
        ingest.ingest_path(path)
    retrieve = RetrieveFlow(
        settings=settings, graph_store=ingest.graph_store, vector_store=ingest.vector_store
    )

    questions = json.loads((FIXTURE_DIR / "questions.json").read_text(encoding="utf-8"))
    ranks: dict[str, int | None] = {}
    for question in questions["questions"]:
        query = question["query"]
        bundle = retrieve.retrieve(query, limit=K, query_type=classify_query(query))
        sources = [_source_of(hit.node_id) for hit in bundle.hits[:K]]
        found = [sources.index(s) + 1 for s in question["expected_sources"] if s in sources]
        ranks[question["id"]] = (
            max(found) if len(found) == len(question["expected_sources"]) else None
        )
    return ranks


def _summary(ranks: dict[str, int | None]) -> dict[str, float]:
    return {
        f"hit_at_{K}": round(sum(rank is not None for rank in ranks.values()) / len(ranks), 4),
        "mrr": round(sum(1 / rank for rank in ranks.values() if rank) / len(ranks), 4),
    }


def _rank_changes(before: dict[str, int | None], after: dict[str, int | None], worse: bool) -> str:
    """'rg-03 1->2, rg-21 4->miss' for questions whose rank moved in the given direction."""

    def key(rank: int | None) -> int:
        return rank or K + 1

    moved = [
        f"{qid} {before[qid] or 'miss'}->{after[qid] or 'miss'}"
        for qid in sorted(before)
        if (key(after[qid]) > key(before[qid])) == worse and after[qid] != before[qid]
    ]
    return ", ".join(moved)


def test_retrieval_ranking_does_not_regress(tmp_path, monkeypatch) -> None:
    # Without this, a dev machine with Ollama up embeds with a real model while CI falls back.
    monkeypatch.setattr(vector_store, "chromadb", None)
    ranks = _ranks(tmp_path)
    summary = _summary(ranks)

    if os.environ.get("RCA_UPDATE_RETRIEVAL_BASELINE") == "1":
        BASELINE_PATH.write_text(
            json.dumps({**summary, "ranks": ranks}, indent=2) + "\n", encoding="utf-8"
        )

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert set(ranks) == set(baseline["ranks"]), (
        "questions.json and baseline.json list different questions; refresh the baseline"
    )
    worse = _rank_changes(baseline["ranks"], ranks, worse=True)
    better = _rank_changes(baseline["ranks"], ranks, worse=False)
    # Any worse rank fails, even when other questions improve: trade-offs need a deliberate
    # baseline refresh so the baseline.json diff shows them in review.
    assert not worse, (
        f"retrieval regression: hit@{K} {baseline[f'hit_at_{K}']:.1%} -> "
        f"{summary[f'hit_at_{K}']:.1%}, MRR {baseline['mrr']:.3f} -> {summary['mrr']:.3f}; "
        f"worse: {worse}; better: {better or 'none'}. "
        "If intended, refresh with RCA_UPDATE_RETRIEVAL_BASELINE=1."
    )
