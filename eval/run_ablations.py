"""Retrieval ablation study — 4 configurations, hit@5 on golden pairs.

Configs
-------
1. vector-only          : VectorStore.query() only
2. vector + FTS         : vector + GraphStore.search_nodes(), no expansion
3. vector + FTS + expand: full RetrieveFlow.retrieve(), no query rewrite
4. full pipeline        : LLM query rewrite + full RetrieveFlow.retrieve()

Metric: hit@5 — fraction of cases where expected_source appears among the
source IDs resolved from the top-5 retrieved hits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rca.config.settings import get_settings
from rca.flows.retrieve_flow import RetrieveFlow, RetrievalHit
from rca.llm.client import ChatMessage, OllamaLLMClient
from rca.store.graph_store import GraphStore
from rca.store.vector_store import VectorStore

CHUNK_SUFFIX = re.compile(r":\d+$")
TOP_K = 5


def resolve_to_source(node_id: str) -> str:
    """Return the parent src: ID for a chunk node, or the ID itself if already src:."""
    if node_id.startswith("src:"):
        return node_id
    base = CHUNK_SUFFIX.sub("", node_id)
    return "src:" + base.split(":", 1)[1]


def hit_at_k(hits: list[RetrievalHit], expected_source: str, k: int = TOP_K) -> bool:
    source_ids = {resolve_to_source(h.node_id) for h in hits[:k]}
    return expected_source in source_ids


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
        cleaned = response.text.strip().split("\n")[0]
        return cleaned if len(cleaned) > 5 else question
    except Exception as exc:
        print(f"  [rewrite] failed: {exc!r} — using original")
        return question


def run_config1(question: str, vector_store: VectorStore, graph_store: GraphStore) -> list[RetrievalHit]:
    """Vector-only: no graph FTS, no expansion."""
    hits = []
    for r in vector_store.query(question, limit=TOP_K):
        node = graph_store.get_node(r.id)
        title = node.title if node else r.metadata.get("title", r.id)
        metadata = node.metadata if node else r.metadata
        hits.append(RetrievalHit(
            node_id=r.id, score=r.score, title=title,
            excerpt=r.document[:240], metadata=metadata,
        ))
    return hits


def run_config2(question: str, vector_store: VectorStore, graph_store: GraphStore) -> list[RetrievalHit]:
    """Vector + FTS merge, no expansion."""
    hit_map: dict[str, RetrievalHit] = {}
    for r in vector_store.query(question, limit=TOP_K):
        node = graph_store.get_node(r.id)
        title = node.title if node else r.metadata.get("title", r.id)
        metadata = node.metadata if node else r.metadata
        current = hit_map.get(r.id)
        if current is None or r.score > current.score:
            hit_map[r.id] = RetrievalHit(
                node_id=r.id, score=r.score, title=title,
                excerpt=r.document[:240], metadata=metadata,
            )
    for node in graph_store.search_nodes(question, limit=TOP_K):
        current = hit_map.get(node.id)
        score = current.score if current else 0.5
        hit_map[node.id] = RetrievalHit(
            node_id=node.id, score=max(score, 0.5),
            title=node.title, excerpt=(node.text or "")[:240],
            metadata=node.metadata,
        )
    return sorted(hit_map.values(), key=lambda h: h.score, reverse=True)[:TOP_K]


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

    config_keys = ["1_vector_only", "2_vector_fts", "3_vector_fts_expand", "4_full_rewrite"]
    config_hits: dict[str, list[bool]] = {k: [] for k in config_keys}
    per_case: list[dict] = []

    for i, pair in enumerate(golden_pairs):
        question = pair["question"]
        expected = pair["expected_source"]
        print(f"[{i+1:02d}/{len(golden_pairs)}] {pair['id']}: {question[:65]}...")

        hits1 = run_config1(question, vector_store, graph_store)
        h1 = hit_at_k(hits1, expected)

        hits2 = run_config2(question, vector_store, graph_store)
        h2 = hit_at_k(hits2, expected)

        bundle3 = retrieve_flow.retrieve(question)
        h3 = hit_at_k(bundle3.hits, expected)

        rewritten = rewrite_query(llm, question)
        bundle4 = retrieve_flow.retrieve(rewritten)
        h4 = hit_at_k(bundle4.hits, expected)

        config_hits["1_vector_only"].append(h1)
        config_hits["2_vector_fts"].append(h2)
        config_hits["3_vector_fts_expand"].append(h3)
        config_hits["4_full_rewrite"].append(h4)

        per_case.append({
            "id": pair["id"],
            "difficulty": pair["difficulty"],
            "expected_source": expected,
            "rewritten_query": rewritten if rewritten != question else None,
            "hit_at_5": {
                "vector_only": h1,
                "vector_fts": h2,
                "vector_fts_expand": h3,
                "full_rewrite": h4,
            },
        })
        print(f"  vector={int(h1)}  +fts={int(h2)}  +expand={int(h3)}  +rewrite={int(h4)}")

    n = len(golden_pairs)
    summary = {k: round(sum(v) / n, 4) for k, v in config_hits.items()}

    print()
    print("=" * 62)
    print(f"{'Retrieval ablations — hit@5 (n=' + str(n) + ')'}")
    print("=" * 62)
    rows = [
        ("1. vector-only",                 "1_vector_only"),
        ("2. vector + FTS",                "2_vector_fts"),
        ("3. vector + FTS + expansion",    "3_vector_fts_expand"),
        ("4. full pipeline (+ rewrite)",   "4_full_rewrite"),
    ]
    for label, key in rows:
        bar = "█" * round(summary[key] * 20)
        print(f"  {label:<35} {summary[key]:.1%}  {bar}")
    print()

    output = {
        "metric": "hit@5",
        "top_k": TOP_K,
        "cases": n,
        "summary": summary,
        "per_case": per_case,
    }
    out_path = Path("eval/results/ablations.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
