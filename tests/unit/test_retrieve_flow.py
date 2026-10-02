from __future__ import annotations

import time
from types import SimpleNamespace

from rca.contracts.nodes import Edge, EdgeKind, Node, NodeKind
from rca.contracts.trace import QueryTrace
from rca.flows.retrieve_flow import RetrievalHit, RetrieveFlow
from rca.store.vector_store import VectorQueryResult


def test_lexical_score_rewards_title_substrings_over_generic_text_overlap() -> None:
    flow = object.__new__(RetrieveFlow)
    query = "What are the two main components of JamPacker and what problem does each solve?"
    query_tokens = flow._tokenize(query)

    jampacker_score = flow._lexical_score(
        query_tokens,
        "Jampacker_An_Efficient_and_Reliable_Robotic_Bin_Packing_System_for_Cuboid_Objects",
        "This paper studies the bin packing problem and explains how each module solves reliability issues.",
    )
    generic_score = flow._lexical_score(
        query_tokens,
        "Comprehensive_Review_of_Robotized_Freight_Packing",
        "This survey covers components, main problem formulations, and how each solver addresses them in packing.",
    )

    assert jampacker_score > generic_score


def test_lexical_score_does_not_match_partial_title_words() -> None:
    flow = object.__new__(RetrieveFlow)
    query_tokens = {"mode"}

    score = flow._lexical_score(
        query_tokens,
        "Vision_Language_Models_A_Survey",
        "",
    )

    assert score == 0.0


def test_lexical_score_does_not_match_partial_text_words() -> None:
    flow = object.__new__(RetrieveFlow)
    query_tokens = {"jampack"}

    score = flow._lexical_score(
        query_tokens,
        "",
        "This review mentions Jampacker systems in passing.",
    )

    assert score == 0.0


def test_expand_to_sources_only_promotes_contains_edges() -> None:
    class StubGraphStore:
        def list_edges(self, node_id: str) -> list[Edge]:
            assert node_id == "chk:pdf/demo:0000"
            return [
                Edge(source="src:pdf/wrong-parent", target=node_id, kind=EdgeKind.references),
                Edge(source="src:pdf/right-parent", target=node_id, kind=EdgeKind.contains),
            ]

        def get_node(self, node_id: str) -> Node | None:
            return Node(
                id=node_id,
                kind=NodeKind.paper,
                title=f"title for {node_id}",
                text="source excerpt",
            )

    flow = object.__new__(RetrieveFlow)
    flow.graph_store = StubGraphStore()

    expanded = flow._expand_to_sources(
        [
            RetrievalHit(
                node_id="chk:pdf/demo:0000",
                score=0.8,
                title="chunk title",
                excerpt="chunk excerpt",
            )
        ]
    )

    assert [hit.node_id for hit in expanded] == ["src:pdf/right-parent"]


def test_retrieve_populates_trace_stages_provenance_and_latency() -> None:
    class StubVectorStore:
        backend_warning = None

        def query(self, query: str, limit: int = 10) -> list[VectorQueryResult]:
            time.sleep(0.001)
            return [
                VectorQueryResult(
                    id="chk:pdf/demo:0000",
                    score=0.91,
                    document="A chunk about JamPacker components.",
                    metadata={"source_id": "src:pdf/demo"},
                )
            ]

    class StubGraphStore:
        def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
            time.sleep(0.001)
            return [
                Node(
                    id="src:pdf/lexical-only",
                    kind=NodeKind.paper,
                    title="Lexical Only Source",
                    text="Lexical terms appear here.",
                )
            ]

        def get_node(self, node_id: str) -> Node | None:
            nodes = {
                "chk:pdf/demo:0000": Node(
                    id="chk:pdf/demo:0000",
                    kind=NodeKind.chunk,
                    title="JamPacker Chunk #1",
                    text="A chunk about JamPacker components.",
                    metadata={"source_id": "src:pdf/demo"},
                ),
                "src:pdf/demo": Node(
                    id="src:pdf/demo",
                    kind=NodeKind.paper,
                    title="JamPacker Source",
                    text="Source preview",
                ),
                "src:pdf/lexical-only": Node(
                    id="src:pdf/lexical-only",
                    kind=NodeKind.paper,
                    title="Lexical Only Source",
                    text="Lexical terms appear here.",
                ),
            }
            return nodes.get(node_id)

        def list_edges(self, node_id: str) -> list[Edge]:
            if node_id == "chk:pdf/demo:0000":
                return [
                    Edge(
                        source="src:pdf/demo",
                        target="chk:pdf/demo:0000",
                        kind=EdgeKind.contains,
                    )
                ]
            return []

    flow = object.__new__(RetrieveFlow)
    flow.graph_store = StubGraphStore()
    flow.vector_store = StubVectorStore()
    trace = QueryTrace(query="JamPacker components")

    bundle = flow.retrieve("JamPacker components", limit=5, trace=trace)

    stage_names = [stage.name for stage in trace.stages]

    assert bundle.trace is trace
    assert stage_names == ["vector_search", "graph_search", "score_merge", "expand_sources"]
    assert all(stage.duration_ms > 0.0 for stage in trace.stages)
    assert trace.total_latency_ms > 0.0
    assert [item.stage for item in trace.provenance] == ["vector", "lexical", "expansion"]


def test_lexical_only_hits_do_not_push_strong_vector_hits_out_of_top_k() -> None:
    class StubVectorStore:
        backend_warning = None

        def query(self, query: str, limit: int = 10) -> list[VectorQueryResult]:
            return [
                VectorQueryResult(
                    id="chk:pdf/wrong-a:0000", score=0.66, document="wrong a", metadata={}
                ),
                VectorQueryResult(
                    id="chk:pdf/wrong-b:0000", score=0.64, document="wrong b", metadata={}
                ),
                VectorQueryResult(
                    id="chk:pdf/wrong-c:0000", score=0.63, document="wrong c", metadata={}
                ),
                VectorQueryResult(
                    id="chk:pdf/wrong-d:0000", score=0.62, document="wrong d", metadata={}
                ),
                VectorQueryResult(
                    id="chk:pdf/correct:0000", score=0.61, document="correct", metadata={}
                ),
            ]

    class StubGraphStore:
        def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
            return [
                Node(
                    id="chk:pdf/lexical-only:0000",
                    kind=NodeKind.chunk,
                    title="Generic scene error node",
                    text="dangerous attribute error observed scene",
                )
            ]

        def get_node(self, node_id: str) -> Node | None:
            return Node(
                id=node_id,
                kind=NodeKind.chunk,
                title=f"title for {node_id}",
                text="chunk text",
                metadata={"source_id": node_id.replace("chk:", "src:").rsplit(":", 1)[0]},
            )

        def list_edges(self, node_id: str) -> list[Edge]:
            return []

    flow = object.__new__(RetrieveFlow)
    flow.graph_store = StubGraphStore()
    flow.vector_store = StubVectorStore()

    bundle = flow.retrieve(
        "What is the most dangerous attribute error observed in the scene?",
        limit=5,
    )

    top_ids = [hit.node_id for hit in bundle.hits[:5]]

    assert "chk:pdf/correct:0000" in top_ids
    assert "chk:pdf/lexical-only:0000" not in top_ids


def test_salient_lexical_only_hits_can_still_enter_top_k() -> None:
    class StubVectorStore:
        backend_warning = None

        def query(self, query: str, limit: int = 10) -> list[VectorQueryResult]:
            return [
                VectorQueryResult(
                    id="chk:pdf/wrong-a:0000", score=0.66, document="wrong a", metadata={}
                ),
                VectorQueryResult(
                    id="chk:pdf/wrong-b:0000", score=0.64, document="wrong b", metadata={}
                ),
                VectorQueryResult(
                    id="chk:pdf/wrong-c:0000", score=0.63, document="wrong c", metadata={}
                ),
                VectorQueryResult(
                    id="chk:pdf/wrong-d:0000", score=0.62, document="wrong d", metadata={}
                ),
            ]

    class StubGraphStore:
        def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
            return [
                Node(
                    id="chk:pdf/lexical-only-generic:0000",
                    kind=NodeKind.chunk,
                    title="Generic scene error node",
                    text="components problem scene",
                ),
                Node(
                    id="chk:pdf/lexical-only-salient:0000",
                    kind=NodeKind.chunk,
                    title="JamPacker system chunk",
                    text="problem details",
                ),
            ]

        def get_node(self, node_id: str) -> Node | None:
            return Node(
                id=node_id,
                kind=NodeKind.chunk,
                title=f"title for {node_id}",
                text="chunk text",
                metadata={"source_id": node_id.replace("chk:", "src:").rsplit(":", 1)[0]},
            )

        def list_edges(self, node_id: str) -> list[Edge]:
            return []

    flow = object.__new__(RetrieveFlow)
    flow.graph_store = StubGraphStore()
    flow.vector_store = StubVectorStore()

    bundle = flow.retrieve(
        "What are the two main components of JamPacker and what problem does each solve?",
        limit=5,
    )

    top_ids = [hit.node_id for hit in bundle.hits[:5]]

    assert "chk:pdf/lexical-only-salient:0000" in top_ids
    assert "chk:pdf/lexical-only-generic:0000" not in top_ids


def test_retrieve_uses_reranker_and_records_stage() -> None:
    class StubVectorStore:
        backend_warning = None

        def query(self, query: str, limit: int = 10) -> list[VectorQueryResult]:
            return [
                VectorQueryResult(id="chk:pdf/a:0000", score=0.91, document="alpha", metadata={}),
                VectorQueryResult(id="chk:pdf/b:0000", score=0.89, document="beta", metadata={}),
                VectorQueryResult(id="chk:pdf/c:0000", score=0.87, document="gamma", metadata={}),
            ]

    class StubGraphStore:
        def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
            return []

        def get_node(self, node_id: str) -> Node | None:
            return Node(
                id=node_id,
                kind=NodeKind.chunk,
                title=f"title for {node_id}",
                text=f"text for {node_id}",
                metadata={},
            )

        def list_edges(self, node_id: str) -> list[Edge]:
            return []

    class StubReranker:
        def rerank(
            self, query: str, hits: list[RetrievalHit], top_k: int | None = None
        ) -> list[RetrievalHit]:
            ordered = sorted(hits, key=lambda hit: hit.node_id, reverse=True)
            keep = len(ordered) if top_k is None else top_k
            return ordered[:keep]

    flow = object.__new__(RetrieveFlow)
    flow.settings = SimpleNamespace(retrieval_fetch_limit=20, reranker_top_k=5)
    flow.graph_store = StubGraphStore()
    flow.vector_store = StubVectorStore()
    flow._reranker = StubReranker()
    trace = QueryTrace(query="demo query")

    bundle = flow.retrieve("demo query", limit=2, trace=trace)

    assert [hit.node_id for hit in bundle.hits] == ["chk:pdf/c:0000", "chk:pdf/b:0000"]
    assert "cross_encoder_rerank" in [stage.name for stage in trace.stages]


def _hit(node_id: str, score: float = 0.5) -> RetrievalHit:
    return RetrievalHit(node_id=node_id, score=score, title=node_id, excerpt="", metadata={})


def test_reserve_new_source_keeps_head_and_reserves_last_slot_for_unseen_source() -> None:
    from rca.flows.retrieve_flow import reserve_new_source

    a = [_hit(f"chk:pdf/a:000{i}") for i in range(5)]
    b = _hit("chk:pdf/b:0000")
    src_a = _hit("src:pdf/a")

    ids = lambda hits: [h.node_id for h in hits]  # noqa: E731
    # one source fills the head: last slot goes to the best-ranked new source
    assert ids(reserve_new_source([*a, b], 5)) == ids([*a[:4], b])
    # a source node of a seen paper counts as the same source
    assert ids(reserve_new_source([*a[:4], src_a, b], 5)) == ids([*a[:4], b])
    # slot n already holds a new source, or no new source anywhere: plain truncation
    assert ids(reserve_new_source([a[0], b, *a[1:]], 5)) == ids([a[0], b, *a[1:4]])
    assert ids(reserve_new_source(a, 5)) == ids(a)
    # head already has two papers but slot n repeats one: still reserved for a third
    c = _hit("chk:pdf/c:0000")
    assert ids(reserve_new_source([a[0], b, a[1], a[2], a[3], c], 5)) == ids(
        [a[0], b, a[1], a[2], c]
    )
    # edge sizes
    assert ids(reserve_new_source([*a, b], 1)) == ids(a[:1])
    assert reserve_new_source([], 5) == []
    assert ids(reserve_new_source(a[:3], 5)) == ids(a[:3])


def test_retrieve_reserves_last_slot_for_new_source_after_rerank() -> None:
    class StubVectorStore:
        backend_warning = None

        def query(self, query: str, limit: int = 10) -> list[VectorQueryResult]:
            ids = [f"chk:pdf/a:000{i}" for i in range(5)] + ["chk:pdf/b:0000"]
            return [
                VectorQueryResult(id=node_id, score=0.9 - 0.01 * rank, document="x", metadata={})
                for rank, node_id in enumerate(ids)
            ]

    class StubGraphStore:
        def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
            return []

        def get_node(self, node_id: str) -> Node | None:
            return None

        def list_edges(self, node_id: str) -> list[Edge]:
            return []

    class ScoreOrderReranker:
        def rerank(self, query, hits, top_k=None):
            ordered = sorted(hits, key=lambda hit: hit.score, reverse=True)
            return ordered if top_k is None else ordered[:top_k]

    def run(reserve: bool) -> list[str]:
        flow = object.__new__(RetrieveFlow)
        flow.settings = SimpleNamespace(
            retrieval_fetch_limit=20, reranker_top_k=10, retrieval_reserve_new_source=reserve
        )
        flow.graph_store = StubGraphStore()
        flow.vector_store = StubVectorStore()
        flow._reranker = ScoreOrderReranker()
        return [hit.node_id for hit in flow.retrieve("demo", limit=5).hits]

    assert run(reserve=True) == [f"chk:pdf/a:000{i}" for i in range(4)] + ["chk:pdf/b:0000"]
    assert run(reserve=False) == [f"chk:pdf/a:000{i}" for i in range(5)]


def test_reserve_new_source_skips_candidates_at_or_below_the_rerank_floor() -> None:
    from rca.flows.retrieve_flow import reserve_new_source

    def scored(node_id: str, rerank: float | None) -> RetrievalHit:
        metadata = {} if rerank is None else {"rerank_score": rerank}
        return RetrievalHit(
            node_id=node_id, score=0.5, title=node_id, excerpt="", metadata=metadata
        )

    head = [scored(f"chk:pdf/a:000{i}", 3.0) for i in range(5)]
    weak_b = scored("chk:pdf/b:0000", -7.0)
    floor_c = scored("chk:pdf/c:0000", -6.0)
    ok_d = scored("chk:pdf/d:0000", -4.3)
    unscored_e = scored("chk:pdf/e:0000", None)

    ids = lambda hits: [h.node_id for h in hits]  # noqa: E731
    # weak and exactly-at-floor candidates are skipped; the next one above the floor wins
    assert ids(reserve_new_source([*head, weak_b, floor_c, ok_d], 5, min_rerank_score=-6.0)) == ids(
        [*head[:4], ok_d]
    )
    # nothing above the floor: plain truncation keeps the same-paper chunk
    assert ids(reserve_new_source([*head, weak_b, floor_c], 5, min_rerank_score=-6.0)) == ids(head)
    # no floor configured: best-ranked new source regardless of score
    assert ids(reserve_new_source([*head, weak_b, ok_d], 5)) == ids([*head[:4], weak_b])
    # hits without a rerank score (reranker off) are not filtered by the floor
    assert ids(reserve_new_source([*head, unscored_e], 5, min_rerank_score=-6.0)) == ids(
        [*head[:4], unscored_e]
    )


def test_retrieve_applies_configured_reserve_floor() -> None:
    class StubVectorStore:
        backend_warning = None

        def query(self, query: str, limit: int = 10) -> list[VectorQueryResult]:
            ids = [f"chk:pdf/a:000{i}" for i in range(5)] + ["chk:pdf/b:0000"]
            return [
                VectorQueryResult(id=node_id, score=0.9 - 0.01 * rank, document="x", metadata={})
                for rank, node_id in enumerate(ids)
            ]

    class StubGraphStore:
        def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
            return []

        def get_node(self, node_id: str) -> Node | None:
            return None

        def list_edges(self, node_id: str) -> list[Edge]:
            return []

    class ScoringReranker:
        """Paper a scores high, paper b scores -7.0 (below the default floor)."""

        def rerank(self, query, hits, top_k=None):
            scored = [
                hit.model_copy(
                    update={"metadata": {"rerank_score": -7.0 if "/b:" in hit.node_id else 3.0}}
                )
                for hit in hits
            ]
            ordered = sorted(scored, key=lambda hit: hit.metadata["rerank_score"], reverse=True)
            return ordered if top_k is None else ordered[:top_k]

    def run(floor: float) -> list[str]:
        flow = object.__new__(RetrieveFlow)
        flow.settings = SimpleNamespace(
            retrieval_fetch_limit=20,
            reranker_top_k=10,
            retrieval_reserve_new_source=True,
            retrieval_reserve_min_rerank_score=floor,
        )
        flow.graph_store = StubGraphStore()
        flow.vector_store = StubVectorStore()
        flow._reranker = ScoringReranker()
        return [hit.node_id for hit in flow.retrieve("demo", limit=5).hits]

    assert run(-6.0) == [f"chk:pdf/a:000{i}" for i in range(5)]
    assert run(-8.0) == [f"chk:pdf/a:000{i}" for i in range(4)] + ["chk:pdf/b:0000"]
