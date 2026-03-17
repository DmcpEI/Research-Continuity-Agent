"""Retrieval workflow built on graph and vector stores."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from rca.config.settings import Settings, get_settings
from rca.contracts.nodes import Edge, EdgeKind
from rca.contracts.trace import HitProvenance, QueryTrace, StageTrace
from rca.retrieval.query_classifier import QueryType
from rca.retrieval.reranker import CrossEncoderReranker
from rca.store.graph_store import GraphStore
from rca.store.vector_store import VectorQueryResult, VectorStore

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9+_.-]*")
TITLE_WORD_PATTERN = re.compile(r"[a-z0-9]+")
LEXICAL_BASE_SCORE = 0.45
LEXICAL_TITLE_WEIGHT = 0.12
LEXICAL_TEXT_WEIGHT = 0.05
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
    "easy",
    "error",
    "most",
    "observed",
    "scene",
    "specific",
}


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
    trace: QueryTrace | None = None


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
        self.vector_store = vector_store or VectorStore(
            self.settings.vector_dir, self.settings.default_collection
        )
        self._reranker: CrossEncoderReranker | None = (
            CrossEncoderReranker(self.settings.reranker_model)
            if self.settings.enable_reranker
            else None
        )

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        trace: QueryTrace | None = None,
        query_type: QueryType | None = None,
    ) -> RetrievalBundle:
        hit_map: dict[str, RetrievalHit] = {}
        hit_stages: dict[str, str] = {}
        related_edges: list[Edge] = []
        query_tokens = self._tokenize(query)
        reranker = getattr(self, "_reranker", None)
        settings = getattr(self, "settings", None)
        fetch_limit = limit
        rerank_limit = limit
        if reranker is not None:
            fetch_limit = max(limit, getattr(settings, "retrieval_fetch_limit", limit))
            rerank_limit = min(limit, getattr(settings, "reranker_top_k", limit))

        if query_type is QueryType.proper_noun:
            lexical_base = 0.65
            vector_scale = 0.70
        elif query_type is QueryType.conceptual:
            lexical_base = 0.30
            vector_scale = 1.00
        else:
            lexical_base = LEXICAL_BASE_SCORE
            vector_scale = 1.00

        vector_started = perf_counter() if trace is not None else None
        vector_results = self.vector_store.query(query, limit=fetch_limit)
        if trace is not None:
            self._append_stage(
                trace,
                name="vector_search",
                started_at=vector_started,
                hit_count=len(vector_results),
                query=query,
            )
            self._append_warning(trace, self.vector_store.backend_warning)

        graph_started = perf_counter() if trace is not None else None
        lexical_nodes = self.graph_store.search_nodes(query, limit=fetch_limit)
        if trace is not None:
            self._append_stage(
                trace,
                name="graph_search",
                started_at=graph_started,
                hit_count=len(lexical_nodes),
                query=query,
            )

        merge_started = perf_counter() if trace is not None else None
        for result in vector_results:
            self._merge_vector_hit(hit_map, result, score_scale=vector_scale)
            hit_stages.setdefault(result.id, "vector")

        for node in lexical_nodes:
            lexical_score = self._lexical_score(
                query_tokens,
                node.title,
                node.text or "",
                base=lexical_base,
            )
            current = hit_map.get(node.id)
            score = current.score if current is not None else lexical_score
            hit_map[node.id] = RetrievalHit(
                node_id=node.id,
                score=max(score, lexical_score),
                title=node.title,
                excerpt=(node.text or "")[:240],
                metadata=node.metadata,
            )
            hit_stages.setdefault(node.id, "lexical")

        ordered_hits = sorted(hit_map.values(), key=lambda h: h.score, reverse=True)[:fetch_limit]
        if trace is not None:
            self._append_stage(
                trace,
                name="score_merge",
                started_at=merge_started,
                hit_count=len(ordered_hits),
                query=query,
            )

        expand_started = perf_counter() if trace is not None else None
        source_hits = self._expand_to_sources(ordered_hits)
        source_hits_new = [h for h in source_hits if h.node_id not in hit_map]
        if trace is not None:
            self._append_stage(
                trace,
                name="expand_sources",
                started_at=expand_started,
                hit_count=len(source_hits_new),
                query=query,
            )

        final_hits = ordered_hits + source_hits_new
        for hit in source_hits_new:
            hit_stages[hit.node_id] = "expansion"

        rerank_started = perf_counter() if trace is not None else None
        if reranker is not None:
            try:
                final_hits = reranker.rerank(query, final_hits, top_k=rerank_limit)
            except Exception as exc:
                final_hits = final_hits[:rerank_limit]
                if trace is not None:
                    self._append_warning(
                        trace,
                        f"cross-encoder reranker unavailable: {type(exc).__name__}: {exc}",
                    )
            if trace is not None:
                self._append_stage(
                    trace,
                    name="cross_encoder_rerank",
                    started_at=rerank_started,
                    hit_count=len(final_hits),
                    query=query,
                )
        else:
            final_hits = final_hits[:limit]

        seen_edge_keys: set[tuple[str, str, str]] = set()
        for hit in final_hits:
            for edge in self.graph_store.list_edges(hit.node_id):
                edge_key = (edge.source, edge.target, edge.kind.value)
                if edge_key not in seen_edge_keys:
                    seen_edge_keys.add(edge_key)
                    related_edges.append(edge)

        if trace is not None:
            trace.provenance = [
                HitProvenance(
                    node_id=hit.node_id,
                    score=hit.score,
                    stage=hit_stages.get(hit.node_id, "expansion"),
                    rank=rank,
                )
                for rank, hit in enumerate(final_hits, start=1)
            ]
            trace.total_latency_ms = sum(stage.duration_ms for stage in trace.stages)

        return RetrievalBundle(
            query=query, hits=final_hits, related_edges=related_edges, trace=trace
        )

    def _merge_vector_hit(
        self,
        hit_map: dict[str, RetrievalHit],
        result: VectorQueryResult,
        score_scale: float = 1.0,
    ) -> None:
        node = self.graph_store.get_node(result.id)
        title = node.title if node else result.metadata.get("title", result.id)
        metadata = node.metadata if node else result.metadata
        excerpt = result.document[:240]
        score = result.score * score_scale

        current = hit_map.get(result.id)
        if current is None or score > current.score:
            hit_map[result.id] = RetrievalHit(
                node_id=result.id,
                score=score,
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
                if edge.target == hit.node_id and edge.kind == EdgeKind.contains:
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

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in TOKEN_PATTERN.findall(text.lower())
            if len(token) >= 3 and token not in STOPWORDS
        }

    def _lexical_score(
        self,
        query_tokens: set[str],
        title: str,
        text: str,
        base: float = LEXICAL_BASE_SCORE,
    ) -> float:
        if not query_tokens:
            return 0.5

        title_word_tokens = set(TITLE_WORD_PATTERN.findall(title.lower()))
        text_word_tokens = set(TITLE_WORD_PATTERN.findall(text.lower()))
        title_overlap = sum(1 for token in query_tokens if token in title_word_tokens)
        text_overlap = sum(1 for token in query_tokens if token in text_word_tokens)

        if title_overlap == 0 and text_overlap == 0:
            return 0.0

        return min(
            0.95,
            base + (LEXICAL_TITLE_WEIGHT * title_overlap) + (LEXICAL_TEXT_WEIGHT * text_overlap),
        )

    @staticmethod
    def _append_stage(
        trace: QueryTrace,
        name: str,
        started_at: float | None,
        hit_count: int,
        query: str,
    ) -> None:
        if started_at is None:
            return
        notes = ""
        if query != trace.query:
            notes = f"query_variant={query}"
        trace.stages.append(
            StageTrace(
                name=name,
                duration_ms=(perf_counter() - started_at) * 1000.0,
                hit_count=hit_count,
                notes=notes,
            )
        )

    @staticmethod
    def _append_warning(trace: QueryTrace, warning: str | None) -> None:
        if warning and warning not in trace.warnings:
            trace.warnings.append(warning)
