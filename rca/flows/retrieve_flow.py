"""Retrieval workflow built on graph and vector stores."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rca.config.settings import Settings, get_settings
from rca.contracts.nodes import Edge
from rca.store.graph_store import GraphStore
from rca.store.vector_store import VectorQueryResult, VectorStore


class RetrievalHit(BaseModel):
    node_id: str
    score: float
    title: str
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalBundle(BaseModel):
    query: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    related_edges: list[Edge] = Field(default_factory=list)


class RetrieveFlow:
    """Compose lexical and vector retrieval results into a single bundle."""

    def __init__(
        self,
        settings: Settings | None = None,
        graph_store: GraphStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.graph_store = graph_store or GraphStore(self.settings.graph_db_path)
        self.vector_store = vector_store or VectorStore(self.settings.vector_dir, self.settings.default_collection)

    def retrieve(self, query: str, limit: int = 10) -> RetrievalBundle:
        hit_map: dict[str, RetrievalHit] = {}
        related_edges: list[Edge] = []

        for result in self.vector_store.query(query, limit=limit):
            self._merge_vector_hit(hit_map, result)

        for node in self.graph_store.search_nodes(query, limit=limit):
            current = hit_map.get(node.id)
            score = current.score if current is not None else 0.5
            hit_map[node.id] = RetrievalHit(
                node_id=node.id,
                score=max(score, 0.5),
                title=node.title,
                excerpt=(node.text or "")[:240],
                metadata=node.metadata,
            )

        ordered_hits = sorted(hit_map.values(), key=lambda h: h.score, reverse=True)[:limit]

        source_hits = self._expand_to_sources(ordered_hits)
        source_hits_new = [h for h in source_hits if h.node_id not in hit_map]

        final_hits = ordered_hits + source_hits_new

        seen_edge_keys: set[tuple[str, str, str]] = set()
        for hit in final_hits:
            for edge in self.graph_store.list_edges(hit.node_id):
                edge_key = (edge.source, edge.target, edge.kind.value)
                if edge_key not in seen_edge_keys:
                    seen_edge_keys.add(edge_key)
                    related_edges.append(edge)

        return RetrievalBundle(query=query, hits=final_hits, related_edges=related_edges)

    def _merge_vector_hit(self, hit_map: dict[str, RetrievalHit], result: VectorQueryResult) -> None:
        node = self.graph_store.get_node(result.id)
        title = node.title if node else result.metadata.get("title", result.id)
        metadata = node.metadata if node else result.metadata
        excerpt = result.document[:240]

        current = hit_map.get(result.id)
        if current is None or result.score > current.score:
            hit_map[result.id] = RetrievalHit(
                node_id=result.id,
                score=result.score,
                title=title,
                excerpt=excerpt,
                metadata=metadata,
            )
    
    def _expand_to_sources(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Follow chunk → source edges to surface parent paper nodes."""
        expanded: dict[str, RetrievalHit] = {}
        
        for hit in hits:
            if hit.node_id.startswith("src:"):
                if hit.node_id not in expanded:
                    expanded[hit.node_id] = hit
                continue
            
            edges = self.graph_store.list_edges(hit.node_id)
            for edge in edges:
                if edge.target == hit.node_id:
                    parent = self.graph_store.get_node(edge.source)
                    if parent and edge.source not in expanded:
                        expanded[edge.source] = RetrievalHit(
                            node_id=parent.id,
                            score=hit.score * 0.95,
                            title=parent.title,
                            excerpt=(parent.text or "")[:240],
                            metadata={**parent.metadata, "expanded_from": hit.node_id},
                        )
        return list(expanded.values())
