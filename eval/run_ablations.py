"""Retrieval ablation study — 4 configurations, hit@5 and hit@10 on golden pairs.

Configs
-------
1. vector-only          : VectorStore.query() only
2. vector + FTS         : vector + GraphStore.search_nodes(), no expansion
3. vector + FTS + expand: full RetrieveFlow.retrieve(), no query rewrite
4. full pipeline        : LLM query rewrite + full RetrieveFlow.retrieve()

Metrics:
  hit@5  — expected_source in top-5 resolved source IDs
  hit@10 — expected_source in top-10 resolved source IDs
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
FETCH_K = 10   # how many results to retrieve per config
HIT_AT = (5, 10)  # report both hit@5 and hit@10


def resolve_to_source(node_id: str) -> str:
    """Return the parent src: ID for a chunk node, or the ID itself if already src:."""
    if node_id.startswith("src:"):
        return node_id
    base = CHUNK_SUFFIX.sub("", node_id)
    return "src:" + base.split(":", 1)[1]


def hit_at_k(hits: list[RetrievalHit], expected_source: str, k: int = 5) -> bool:
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
    for r in vector_store.query(question, limit=FETCH_K):
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

    config_keys = ["1_vector_only", "2_vector_fts", "3_vector_fts_expand", "4_full_rewrite"]
    # track hits at both k=5 and k=10
    hits5: dict[str, list[bool]] = {k: [] for k in config_keys}
    hits10: dict[str, list[bool]] = {k: [] for k in config_keys}
    per_case: list[dict] = []

    for i, pair in enumerate(golden_pairs):
        question = pair["question"]
        expected = pair["expected_source"]
        print(f"[{i+1:02d}/{len(golden_pairs)}] {pair['id']}: {question[:65]}...")

        hits1 = run_config1(question, vector_store, graph_store)
        hits2 = run_config2(question, vector_store, graph_store)
        bundle3 = retrieve_flow.retrieve(question)
        rewritten = rewrite_query(llm, question)
        bundle4 = retrieve_flow.retrieve(rewritten)

        # flag jampacker-001 specifically
        if pair["id"] == "jampacker-001":
            print(f"  [jampacker-001 rewrite] → {rewritten!r}")

        all_hits = [hits1, hits2, bundle3.hits, bundle4.hits]
        for key, hit_list in zip(config_keys, all_hits):
            hits5[key].append(hit_at_k(hit_list, expected, k=5))
            hits10[key].append(hit_at_k(hit_list, expected, k=10))

        per_case.append({
            "id": pair["id"],
            "difficulty": pair["difficulty"],
            "expected_source": expected,
            "rewritten_query": rewritten if rewritten != question else None,
            "hit_at_5":  {k: hits5[k][-1]  for k in config_keys},
            "hit_at_10": {k: hits10[k][-1] for k in config_keys},
        })
        h5  = [int(hits5[k][-1])  for k in config_keys]
        h10 = [int(hits10[k][-1]) for k in config_keys]
        print(f"  @5  vector={h5[0]}  +fts={h5[1]}  +expand={h5[2]}  +rewrite={h5[3]}")
        print(f"  @10 vector={h10[0]}  +fts={h10[1]}  +expand={h10[2]}  +rewrite={h10[3]}")

    n = len(golden_pairs)
    summary5  = {k: round(sum(v) / n, 4) for k, v in hits5.items()}
    summary10 = {k: round(sum(v) / n, 4) for k, v in hits10.items()}

    rows = [
        ("1. vector-only",                 "1_vector_only"),
        ("2. vector + FTS",                "2_vector_fts"),
        ("3. vector + FTS + expansion",    "3_vector_fts_expand"),
        ("4. full pipeline (+ rewrite)",   "4_full_rewrite"),
    ]

    print()
    print("=" * 68)
    print(f"Retrieval ablations — n={n}")
    print(f"{'Configuration':<35}  {'hit@5':>6}  {'hit@10':>7}")
    print("-" * 68)
    for label, key in rows:
        print(f"  {label:<33}  {summary5[key]:>6.1%}  {summary10[key]:>7.1%}")
    print("=" * 68)
    print()

    output = {
        "fetch_k": FETCH_K,
        "cases": n,
        "summary_hit_at_5":  summary5,
        "summary_hit_at_10": summary10,
        "per_case": per_case,
    }
    out_path = Path("eval/results/ablations.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
