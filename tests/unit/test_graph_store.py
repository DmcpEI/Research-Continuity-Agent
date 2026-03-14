from __future__ import annotations

from rca.contracts.ids import make_chunk_id, make_source_id
from rca.contracts.nodes import Edge, EdgeKind, Node, NodeKind
from rca.store.graph_store import GraphStore


def test_graph_store_upserts_and_queries_nodes(tmp_path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")

    source_id = make_source_id("note", "Store Test")
    chunk_id = make_chunk_id(source_id, 0)

    store.upsert_node(Node(id=source_id, kind=NodeKind.note, title="Store Test", text="research continuity"))
    store.upsert_node(Node(id=chunk_id, kind=NodeKind.chunk, title="Store Test #1", text="continuity memory"))
    store.upsert_edge(Edge(source=source_id, target=chunk_id, kind=EdgeKind.contains))

    node = store.get_node(source_id)
    hits = store.search_nodes("continuity")
    edges = store.list_edges(source_id)

    assert node is not None
    assert node.title == "Store Test"
    assert any(hit.id == chunk_id for hit in hits)
    assert len(edges) == 1


def test_graph_store_tokenizes_multiword_queries(tmp_path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")

    source_id = make_source_id("pdf", "jampacker-paper")
    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="Jampacker Paper",
            text="The Fault Recovery Module improves reliability in JamPacker.",
        )
    )

    hits = store.search_nodes("What does the Fault Recovery Module do in JamPacker?")

    assert hits
    assert hits[0].id == source_id


def test_graph_store_like_underscore_is_literal(tmp_path) -> None:
    """_ in a search token must match a literal underscore, not any character."""
    store = GraphStore(tmp_path / "graph.sqlite3")

    exact_id = make_source_id("pdf", "bert-base-uncased")
    wild_id = make_source_id("pdf", "bertx-base-uncased")

    store.upsert_node(Node(id=exact_id, kind=NodeKind.paper, title="bert_base_uncased model", text=None))
    store.upsert_node(Node(id=wild_id, kind=NodeKind.paper, title="bertXbase uncased model", text=None))

    hits = store.search_nodes("bert_base_uncased")
    hit_ids = [h.id for h in hits]

    assert exact_id in hit_ids, "exact underscore title should match"
    assert wild_id not in hit_ids, "_ in token must not act as SQL wildcard matching bertXbase"
