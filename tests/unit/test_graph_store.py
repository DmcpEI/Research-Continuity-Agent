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
