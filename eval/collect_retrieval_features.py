"""Collect retrieval-only features for abstention calibration.

Mirrors the retrieval half of GenerateFlow.generate_answer (classify, rewrite,
retrieve, raw-query fallback) without answer generation, and writes one row of
retrieval signals per golden question.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca.contracts.trace import QueryTrace  # noqa: E402
from rca.flows.generate_flow import GenerateFlow  # noqa: E402
from rca.retrieval.query_classifier import QueryType, classify_query  # noqa: E402

LIMIT = 5  # GenerateFlow.generate_answer default


def to_source_id(node_id: str) -> str:
    if node_id.startswith("chk:"):
        return "src:" + node_id.rsplit(":", 1)[0].split(":", 1)[1]
    return node_id


def collect_row(flow: GenerateFlow, pair: dict) -> dict:
    started = perf_counter()
    query = pair["question"]
    query_type = classify_query(query)
    row = {
        "id": pair["id"],
        "answerable": pair.get("answerable", True),
        "negative_type": pair.get("negative_type"),
        "category": pair.get("category"),
        "difficulty": pair["difficulty"],
        "query_type": query_type.value,
    }
    if query_type is QueryType.conversational:
        return row | {"skipped": "conversational"}

    trace = QueryTrace(query=query)
    rewritten = query if query_type is QueryType.proper_noun else flow._rewrite_query(query, trace)
    bundle = flow.retrieve_flow.retrieve(rewritten, limit=LIMIT, trace=trace, query_type=query_type)
    used_fallback = False
    if bundle.hits and not flow._select_context_hits(bundle) and rewritten != query:
        bundle = flow.retrieve_flow.retrieve(query, limit=LIMIT, trace=trace, query_type=query_type)
        used_fallback = True

    scores = [hit.score for hit in bundle.hits]
    ce = [hit.metadata["rerank_score"] for hit in bundle.hits if "rerank_score" in hit.metadata]
    sources = [to_source_id(hit.node_id) for hit in bundle.hits]
    expected = set(pair.get("expected_sources") or []) | (
        {pair["expected_source"]} if pair.get("expected_source") else set()
    )
    stage_max: dict[str, float] = {}
    for prov in trace.provenance:
        stage_max[prov.stage] = max(stage_max.get(prov.stage, float("-inf")), prov.score)

    return row | {
        "rewritten": rewritten != query,
        "used_fallback": used_fallback,
        "n_hits": len(scores),
        "scores": scores,
        "max_score": max(scores, default=0.0),
        "mean_score": mean(scores) if scores else 0.0,
        "score_gap": scores[0] - scores[1] if len(scores) > 1 else (scores[0] if scores else 0.0),
        "score_spread": max(scores) - min(scores) if scores else 0.0,
        "rerank_scores": ce,
        "rerank_max": max(ce, default=None),
        "rerank_mean": mean(ce) if ce else None,
        "rerank_gap": ce[0] - ce[1] if len(ce) > 1 else None,
        "hits_above_050": sum(s >= 0.50 for s in scores),
        "hits_above_055": sum(s > 0.55 for s in scores),
        "context_hits": len(flow._select_context_hits(bundle)) if bundle.hits else 0,
        "unique_sources": len(set(sources)),
        "top_source_share": sources.count(sources[0]) / len(sources) if sources else 0.0,
        "top_stage": trace.provenance[0].stage if trace.provenance else None,
        "stage_max_scores": stage_max,
        "query_len_tokens": len(query.split()),
        "expected_source_hit": bool(expected & set(sources)) if expected else None,
        "latency_ms": (perf_counter() - started) * 1000.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-path", default="eval/golden.json")
    parser.add_argument("--output-dir", default="eval/results")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.golden_path).read_text(encoding="utf-8"))
    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    flow = GenerateFlow()

    rows = []
    for index, pair in enumerate(pairs, 1):
        rows.append(collect_row(flow, pair))
        print(f"[{index}/{len(pairs)}] {pair['id']} max={rows[-1].get('max_score', '-')}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir) / f"retrieval_features_{stamp}.json"
    output.write_text(
        json.dumps(
            {
                "created_at": stamp,
                "golden_path": args.golden_path,
                "rewrite_model": getattr(flow.rewrite_llm, "model", ""),
                "embedding_model": flow.settings.embedding_model,
                "limit": LIMIT,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
