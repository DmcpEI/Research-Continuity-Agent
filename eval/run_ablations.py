"""Retrieval ablation study — 5 configurations, hit@5 and hit@10 on golden pairs.

Configs
-------
0. fts5-only            : GraphStore.search_nodes_fts5() only
1. vector-only          : VectorStore.query() only
2. vector + keyword     : vector + GraphStore.search_nodes(), no expansion
3. vector + keyword + expand: full RetrieveFlow.retrieve(), no query rewrite
4. full pipeline        : LLM query rewrite + full RetrieveFlow.retrieve()

Metrics:
  hit@5  — all expected sources in top-5 resolved source IDs
  hit@10 — all expected sources in top-10 resolved source IDs
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rca.config.settings import get_settings
from rca.flows.generate_flow import GenerateFlow
from rca.flows.retrieve_flow import RetrieveFlow, RetrievalHit
from rca.llm.client import ChatMessage, OllamaLLMClient
from rca.store.graph_store import GraphStore
from rca.store.vector_store import VectorStore

CHUNK_SUFFIX = re.compile(r":\d+$")
FETCH_K = 10   # how many results to retrieve per config
HIT_AT = (5, 10)  # report both hit@5 and hit@10


def resolve_to_source(node_id: str) -> str:
    """Return the parent src: ID for a chunk node, or the ID itself if already src:."""
    if node_id.startswith("src:"):
        return node_id
    base = CHUNK_SUFFIX.sub("", node_id)
    return "src:" + base.split(":", 1)[1]


def expected_source_ids(pair: dict[str, Any]) -> list[str]:
    sources = [source for source in pair.get("expected_sources", []) if source]
    if sources:
        return sources
    source = pair.get("expected_source")
    return [source] if source else []


def is_retrieval_case(pair: dict[str, Any]) -> bool:
    return pair.get("answerable", True) and bool(expected_source_ids(pair))


def hit_at_k(hits: list[RetrievalHit], expected_sources: list[str], k: int = 5) -> bool:
    source_ids = {resolve_to_source(h.node_id) for h in hits[:k]}
    return all(source in source_ids for source in expected_sources)


def summarize_subset(cases: list[dict[str, Any]], config_keys: list[str]) -> dict[str, Any]:
    total = len(cases)
    if total == 0:
        return {
            "cases": 0,
            "summary_hit_at_5": {key: 0.0 for key in config_keys},
            "summary_hit_at_10": {key: 0.0 for key in config_keys},
        }
    return {
        "cases": total,
        "summary_hit_at_5": {
            key: round(sum(case["hit_at_5"][key] for case in cases) / total, 4) for key in config_keys
        },
        "summary_hit_at_10": {
            key: round(sum(case["hit_at_10"][key] for case in cases) / total, 4) for key in config_keys
        },
    }


def build_bucket_summary(
    per_case: list[dict[str, Any]], config_keys: list[str], field: str, ordered_values: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    evaluated = [case for case in per_case if not case["skipped"]]
    if ordered_values is None:
        ordered_values = []
        seen: set[str] = set()
        for case in evaluated:
            value = case.get(field)
            if not value or value in seen:
                continue
            seen.add(value)
            ordered_values.append(value)

    summary: dict[str, dict[str, Any]] = {}
    for value in ordered_values:
        subset = [case for case in evaluated if case.get(field) == value]
        if not subset:
            continue
        summary[value] = summarize_subset(subset, config_keys)
    return summary


def print_bucket_table(
    title: str,
    summary: dict[str, dict[str, Any]],
    rows: list[tuple[str, str]],
    left_header: str,
) -> None:
    print("=" * 140)
    print(title)
    print(
        f"{left_header:<18} {'n':>3}  "
        f"{'fts5@5':>8} {'fts5@10':>9}  "
        f"{'dense@5':>9} {'dense@10':>10}  "
        f"{'+like@5':>8} {'+like@10':>9}  "
        f"{'+expand@5':>9} {'+expand@10':>10}  "
        f"{'+rewrite@5':>10} {'+rewrite@10':>11}"
    )
    print("-" * 140)
    for label, key in rows:
        metrics = summary.get(key)
        if metrics is None:
            continue
        hit5 = metrics["summary_hit_at_5"]
        hit10 = metrics["summary_hit_at_10"]
        print(
            f"  {label:<16} {metrics['cases']:>3}  "
            f"{hit5['0_fts5_only']:>8.1%} {hit10['0_fts5_only']:>9.1%}  "
            f"{hit5['1_vector_only']:>9.1%} {hit10['1_vector_only']:>10.1%}  "
            f"{hit5['2_vector_keyword']:>8.1%} {hit10['2_vector_keyword']:>9.1%}  "
            f"{hit5['3_vector_keyword_expand']:>9.1%} {hit10['3_vector_keyword_expand']:>10.1%}  "
            f"{hit5['4_full_rewrite']:>10.1%} {hit10['4_full_rewrite']:>11.1%}"
        )
    print("=" * 140)
    print()


def rewrite_query(llm: OllamaLLMClient, question: str) -> str:
    try:
        messages = [
            ChatMessage(
                role="user",
                content=(
                    "Convert this research question into a dense technical search query of 8-12 keywords. "
                    "Include domain-specific terms, method names, and technical concepts. "
                    "No explanation, no punctuation, no full sentences. Just keywords.\n\n"
                    f"Question: {question}\n\nKeywords:"
                ),
            )
        ]
        response = llm.chat(messages)
        return GenerateFlow.sanitize_rewritten_query(question, response.text)
    except Exception as exc:
        print(f"  [rewrite] failed: {exc!r} — using original")
        return question


def run_config1(question: str, vector_store: VectorStore, graph_store: GraphStore) -> list[RetrievalHit]:
    """Vector-only: no graph keyword search, no expansion."""
    hits = []
    for r in vector_store.query(question, limit=FETCH_K):
        node = graph_store.get_node(r.id)
        title = node.title if node else r.metadata.get("title", r.id)
        metadata = node.metadata if node else r.metadata
        hits.append(RetrievalHit(
            node_id=r.id, score=r.score, title=title,
            excerpt=r.document[:240], metadata=metadata,
        ))
    return hits


def run_config0(question: str, graph_store: GraphStore) -> list[RetrievalHit]:
    """FTS5-only BM25 baseline: lexical search without vector, expansion, or rewrite."""
    hits = []
    for rank, node in enumerate(graph_store.search_nodes_fts5(question, limit=FETCH_K), start=1):
        hits.append(
            RetrievalHit(
                node_id=node.id,
                score=max(0.0, 1.0 - ((rank - 1) * 0.01)),
                title=node.title,
                excerpt=(node.text or "")[:240],
                metadata=node.metadata,
            )
        )
    return hits


def run_config2(question: str, vector_store: VectorStore, graph_store: GraphStore) -> list[RetrievalHit]:
    """Vector + keyword-search merge, no expansion."""
    hit_map: dict[str, RetrievalHit] = {}
    for r in vector_store.query(question, limit=FETCH_K):
        node = graph_store.get_node(r.id)
        title = node.title if node else r.metadata.get("title", r.id)
        metadata = node.metadata if node else r.metadata
        current = hit_map.get(r.id)
        if current is None or r.score > current.score:
            hit_map[r.id] = RetrievalHit(
                node_id=r.id, score=r.score, title=title,
                excerpt=r.document[:240], metadata=metadata,
            )
    for node in graph_store.search_nodes(question, limit=FETCH_K):
        current = hit_map.get(node.id)
        score = current.score if current else 0.5
        hit_map[node.id] = RetrievalHit(
            node_id=node.id, score=max(score, 0.5),
            title=node.title, excerpt=(node.text or "")[:240],
            metadata=node.metadata,
        )
    return sorted(hit_map.values(), key=lambda h: h.score, reverse=True)[:FETCH_K]


def main() -> None:
    settings = get_settings()
    graph_store = GraphStore(settings.graph_db_path)
    vector_store = VectorStore(settings.vector_dir, settings.default_collection)
    retrieve_flow = RetrieveFlow(
        settings=settings, graph_store=graph_store, vector_store=vector_store
    )
    llm = OllamaLLMClient(
        base_url=settings.embedding_base_url,
        model=settings.generation_model,
    )

    raw = json.loads(Path("eval/golden.json").read_text(encoding="utf-8"))
    golden_pairs = raw.get("pairs", raw) if isinstance(raw, dict) else raw

    print(f"Loaded {len(golden_pairs)} golden pairs | vector backend: {vector_store.backend}")
    print()

    config_keys = [
        "0_fts5_only",
        "1_vector_only",
        "2_vector_keyword",
        "3_vector_keyword_expand",
        "4_full_rewrite",
    ]
    # track hits at both k=5 and k=10
    hits5: dict[str, list[bool]] = {k: [] for k in config_keys}
    hits10: dict[str, list[bool]] = {k: [] for k in config_keys}
    per_case: list[dict] = []

    evaluated_cases = 0

    for i, pair in enumerate(golden_pairs):
        question = pair["question"]
        expected = expected_source_ids(pair)
        print(f"[{i+1:02d}/{len(golden_pairs)}] {pair['id']}: {question[:65]}...")

        if not is_retrieval_case(pair):
            skip_reason = "unanswerable" if not pair.get("answerable", True) else "missing expected source"
            per_case.append({
                "id": pair["id"],
                "difficulty": pair["difficulty"],
                "category": pair.get("category"),
                "bucket": pair.get("category"),
                "expected_source": pair.get("expected_source"),
                "expected_sources": expected,
                "skipped": True,
                "skip_reason": skip_reason,
            })
            print(f"  skipped ({skip_reason})")
            continue

        evaluated_cases += 1

        hits0 = run_config0(question, graph_store)
        hits1 = run_config1(question, vector_store, graph_store)
        hits2 = run_config2(question, vector_store, graph_store)
        bundle3 = retrieve_flow.retrieve(question)
        rewritten = rewrite_query(llm, question)
        bundle4 = retrieve_flow.retrieve(rewritten)

        # flag jampacker-001 specifically
        if pair["id"] == "jampacker-001":
            print(f"  [jampacker-001 rewrite] → {rewritten!r}")

        all_hits = [hits0, hits1, hits2, bundle3.hits, bundle4.hits]
        for key, hit_list in zip(config_keys, all_hits):
            hits5[key].append(hit_at_k(hit_list, expected, k=5))
            hits10[key].append(hit_at_k(hit_list, expected, k=10))

        per_case.append({
            "id": pair["id"],
            "difficulty": pair["difficulty"],
            "category": pair.get("category"),
            "bucket": pair.get("category"),
            "expected_source": pair.get("expected_source"),
            "expected_sources": expected,
            "rewritten_query": rewritten if rewritten != question else None,
            "hit_at_5":  {k: hits5[k][-1]  for k in config_keys},
            "hit_at_10": {k: hits10[k][-1] for k in config_keys},
            "skipped": False,
        })
        h5  = [int(hits5[k][-1])  for k in config_keys]
        h10 = [int(hits10[k][-1]) for k in config_keys]
        print(f"  @5  fts5={h5[0]}  dense={h5[1]}  +like={h5[2]}  +expand={h5[3]}  +rewrite={h5[4]}")
        print(f"  @10 fts5={h10[0]}  dense={h10[1]}  +like={h10[2]}  +expand={h10[3]}  +rewrite={h10[4]}")

    n = evaluated_cases
    if n == 0:
        summary5 = {k: 0.0 for k in config_keys}
        summary10 = {k: 0.0 for k in config_keys}
    else:
        summary5 = {k: round(sum(v) / n, 4) for k, v in hits5.items()}
        summary10 = {k: round(sum(v) / n, 4) for k, v in hits10.items()}

    rows = [
        ("0. fts5-only (BM25 baseline)",       "0_fts5_only"),
        ("1. vector-only (dense baseline)",    "1_vector_only"),
        ("2. vector + keyword (LIKE)",         "2_vector_keyword"),
        ("3. vector + keyword + expansion",    "3_vector_keyword_expand"),
        ("4. full pipeline (+ rewrite)",       "4_full_rewrite"),
    ]
    category_summary = build_bucket_summary(per_case, config_keys, "category")
    difficulty_summary = build_bucket_summary(
        per_case,
        config_keys,
        "difficulty",
        ordered_values=["easy", "medium", "hard"],
    )

    print()
    print("=" * 74)
    print(f"Retrieval ablations — n={n} evaluated / {len(golden_pairs)} loaded")
    print(f"{'Configuration':<39}  {'hit@5':>6}  {'hit@10':>7}")
    print("-" * 74)
    for label, key in rows:
        print(f"  {label:<37}  {summary5[key]:>6.1%}  {summary10[key]:>7.1%}")
    print("=" * 74)
    print()

    print_bucket_table(
        "Retrieval ablations by category",
        category_summary,
        [(key.replace("_", " "), key) for key in category_summary.keys()],
        "Category",
    )
    print_bucket_table(
        "Retrieval ablations by difficulty",
        difficulty_summary,
        [(key, key) for key in ["easy", "medium", "hard"]],
        "Difficulty",
    )

    output = {
        "fetch_k": FETCH_K,
        "loaded_cases": len(golden_pairs),
        "cases": n,
        "skipped_cases": len(golden_pairs) - n,
        "summary_hit_at_5":  summary5,
        "summary_hit_at_10": summary10,
        "summary_by_category": category_summary,
        "summary_by_difficulty": difficulty_summary,
        "per_case": per_case,
    }
    out_path = Path("eval/results/ablations.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
