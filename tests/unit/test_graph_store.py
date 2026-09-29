from __future__ import annotations

from rca.contracts.ids import make_chunk_id, make_source_id
from rca.contracts.nodes import Edge, EdgeKind, Node, NodeKind
from rca.contracts.revisions import IngestStatus, SourceRevision
from rca.store.graph_store import GraphStore


def test_graph_store_upserts_and_queries_nodes(tmp_path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")

    source_id = make_source_id("note", "Store Test")
    chunk_id = make_chunk_id(source_id, 0)

    store.upsert_node(
        Node(id=source_id, kind=NodeKind.note, title="Store Test", text="research continuity")
    )
    store.upsert_node(
        Node(id=chunk_id, kind=NodeKind.chunk, title="Store Test #1", text="continuity memory")
    )
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

    store.upsert_node(
        Node(id=exact_id, kind=NodeKind.paper, title="bert_base_uncased model", text=None)
    )
    store.upsert_node(
        Node(id=wild_id, kind=NodeKind.paper, title="bertXbase uncased model", text=None)
    )

    hits = store.search_nodes("bert_base_uncased")
    hit_ids = [h.id for h in hits]

    assert exact_id in hit_ids, "exact underscore title should match"
    assert wild_id not in hit_ids, "_ in token must not act as SQL wildcard matching bertXbase"


def test_graph_store_fts5_returns_ranked_results(tmp_path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")

    source_id = make_source_id("pdf", "fault-recovery-paper")
    distractor_id = make_source_id("pdf", "irrelevant-paper")

    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="Fault Recovery Module for JamPacker",
            text="The fault recovery module improves execution reliability.",
        )
    )
    store.upsert_node(
        Node(
            id=distractor_id,
            kind=NodeKind.paper,
            title="Unrelated Planning Paper",
            text="This paper discusses scene graphs and planning.",
        )
    )

    hits = store.search_nodes("fault recovery module", limit=5)

    assert hits
    assert hits[0].id == source_id
    assert distractor_id not in [hit.id for hit in hits[:1]]


def test_graph_store_like_search_remains_available_for_reference(tmp_path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")

    source_id = make_source_id("pdf", "like-reference-paper")
    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="Reference LIKE Search Paper",
            text="This text mentions fault recovery and jam packing.",
        )
    )

    hits = store.search_nodes_like("fault recovery", limit=5)

    assert hits
    assert hits[0].id == source_id


def test_graph_store_create_schema_populates_fts_index(tmp_path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    source_id = make_source_id("pdf", "fts-rebuild-paper")
    store.upsert_node(
        Node(
            id=source_id,
            kind=NodeKind.paper,
            title="FTS Rebuild Paper",
            text="The rebuild path should repopulate the fts table.",
        )
    )

    with store.connect() as connection:
        connection.execute("DELETE FROM nodes_fts")

    store.create_schema()

    with store.connect() as connection:
        fts_rows = connection.execute("SELECT count(*) FROM nodes_fts").fetchone()[0]

    assert fts_rows == 1


def test_graph_store_revision_lookups_and_chunk_cleanup(tmp_path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    source_id = make_source_id("note", "Versioned Note")
    chunk_ids = [make_chunk_id(source_id, 0), make_chunk_id(source_id, 1)]

    store.upsert_node(Node(id=source_id, kind=NodeKind.note, title="Versioned", text="alpha"))
    for chunk_id in chunk_ids:
        store.upsert_node(
            Node(id=chunk_id, kind=NodeKind.chunk, title=chunk_id, text="chunk content")
        )
        store.upsert_edge(Edge(source=source_id, target=chunk_id, kind=EdgeKind.contains))

    first = SourceRevision(
        source_id=source_id,
        revision_number=store.next_revision_number(source_id),
        ingest_status=IngestStatus.created,
        path="/tmp/versioned-note.md",
        ingest_name="versioned-note",
        title="Versioned",
        file_sha256="file-a",
        content_sha256="content-a",
    )
    store.insert_source_revision(first)

    second = SourceRevision(
        source_id=source_id,
        revision_number=store.next_revision_number(source_id),
        ingest_status=IngestStatus.unchanged,
        path="/tmp/renamed-note.md",
        ingest_name="renamed-note",
        title="Renamed",
        file_sha256="file-b",
        content_sha256="content-a",
    )
    store.insert_source_revision(second)

    latest = store.find_latest_revision_for_source(source_id)
    by_file = store.find_revision_by_file_hash("file-a")
    by_content = store.find_revision_by_content_hash("content-a")
    by_name = store.find_revision_by_ingest_name("renamed-note")
    deleted = store.delete_source_chunks(source_id)

    assert latest is not None
    assert latest.revision_number == 2
    assert by_file is not None
    assert by_file.revision_id == first.revision_id
    assert by_content is not None
    assert by_content.revision_id == second.revision_id
    assert by_name is not None
    assert by_name.revision_id == second.revision_id
    assert deleted == chunk_ids
    assert store.list_source_chunk_ids(source_id) == []
    assert store.get_node(chunk_ids[0]) is None


def test_graph_store_replace_source_chunks_atomic_is_consistent(tmp_path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    source_id = make_source_id("note", "Atomic Test")
    old_chunk_ids = [make_chunk_id(source_id, index) for index in range(3)]

    store.upsert_node(Node(id=source_id, kind=NodeKind.note, title="Atomic", text="v1"))
    for chunk_id in old_chunk_ids:
        store.upsert_node(Node(id=chunk_id, kind=NodeKind.chunk, title=chunk_id, text="old"))
        store.upsert_edge(Edge(source=source_id, target=chunk_id, kind=EdgeKind.contains))

    new_source = Node(id=source_id, kind=NodeKind.note, title="Atomic", text="v2")
    new_chunk_ids = [make_chunk_id(source_id, index) for index in range(2)]
    new_nodes = [
        Node(id=chunk_id, kind=NodeKind.chunk, title=chunk_id, text="new")
        for chunk_id in new_chunk_ids
    ]
    new_edges = [
        Edge(source=source_id, target=chunk_id, kind=EdgeKind.contains)
        for chunk_id in new_chunk_ids
    ]

    returned_old = store.replace_source_chunks_atomic(source_id, new_source, new_nodes, new_edges)

    assert set(returned_old) == set(old_chunk_ids)
    assert store.list_source_chunk_ids(source_id) == new_chunk_ids
    for old_chunk_id in set(old_chunk_ids) - set(new_chunk_ids):
        assert store.get_node(old_chunk_id) is None
    for new_chunk_id in new_chunk_ids:
        assert store.get_node(new_chunk_id) is not None
    latest_source = store.get_node(source_id)
    assert latest_source is not None
    assert latest_source.text == "v2"


def test_graph_store_replace_source_chunks_atomic_handles_empty_existing(tmp_path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    source_id = make_source_id("note", "Fresh")
    source_node = Node(id=source_id, kind=NodeKind.note, title="Fresh", text="body")
    chunk_id = make_chunk_id(source_id, 0)
    chunk_node = Node(id=chunk_id, kind=NodeKind.chunk, title="Fresh #1", text="body")
    edge = Edge(source=source_id, target=chunk_id, kind=EdgeKind.contains)

    old_ids = store.replace_source_chunks_atomic(source_id, source_node, [chunk_node], [edge])

    assert old_ids == []
    assert store.list_source_chunk_ids(source_id) == [chunk_id]
