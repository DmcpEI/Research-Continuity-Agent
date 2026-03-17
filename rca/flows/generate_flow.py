"""Generation flow — grounded answers with citation enforcement."""

from __future__ import annotations

import re
from time import perf_counter

from pydantic import BaseModel, Field

from rca.config.settings import Settings, get_settings
from rca.contracts.trace import QueryTrace, StageTrace
from rca.flows.retrieve_flow import RetrieveFlow, RetrievalBundle
from rca.llm.client import ChatMessage, EchoLLMClient, LLMClient
from rca.retrieval.query_classifier import QueryType, classify_query


class Citation(BaseModel):
    source_id: str
    title: str
    excerpt: str


class GeneratedAnswer(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = False
    abstained: bool = False
    trace: QueryTrace | None = None


class GenerateFlow:
    """Generate grounded answers from retrieved context."""

    SYSTEM_PROMPT = """You are a research assistant with access to a personal knowledge base.
Your job is to answer questions using ONLY the context provided below.
Rules:
- If the context contains enough information to answer, every factual claim MUST be followed immediately by [[source_id]] where source_id is the EXACT ID shown in square brackets at the start of the relevant context block
- Use the full ID as written, e.g. [[src:pdf/paper_name]] or [[chk:pdf/paper_name:0012]]
- If the context does not contain enough information, say so clearly and do not include any [[...]] citations
- Never invent facts, authors, or results not present in the context
- Be concise and precise — this is a research tool, not a chatbot"""
    _ABSTENTION_SIGNALS = (
        "does not contain information",
        "no information",
        "cannot find",
        "not mentioned in",
        "no relevant information",
        "cannot answer",
        "not provided in",
    )
    _CITATION_PATTERN = re.compile(r"\[\[((?:src|chk):[^\]]+)\]\]")
    _ABSTENTION_SCORE_THRESHOLD = 0.50
    _REWRITE_PROMPT_TEMPLATE = (
        "Return 1 to 3 short additional search keywords or phrases that would improve retrieval "
        "for this research question. Keep exact method names or acronyms only if helpful. "
        "Do not invent terms. Do not concatenate words. Output only keywords, separated by spaces.\n\n"
        "Question: {question}\n\nKeywords:"
    )

    def __init__(
        self,
        settings: Settings | None = None,
        retrieve_flow: RetrieveFlow | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieve_flow = retrieve_flow or RetrieveFlow()

        if llm_client is not None:
            self.llm = llm_client
        elif self.settings.generation_model:
            from rca.llm.client import OllamaLLMClient
            self.llm = OllamaLLMClient(
                base_url=self.settings.llm_base_url,
                model=self.settings.generation_model,
                api_key=self.settings.llm_api_key,
            )
        else:
            self.llm = EchoLLMClient()

    def generate_answer(self, query: str, limit: int = 5, trace: QueryTrace | None = None) -> GeneratedAnswer:
        trace = trace or QueryTrace(query=query)
        trace.model = getattr(self.llm, "model", self.llm.__class__.__name__)
        query_type = classify_query(query)
        trace.query_type = query_type.value

        # Step 1: retrieve grounded context
        if query_type is QueryType.proper_noun:
            rewritten = query
            self._append_warning(trace, "rewrite skipped: proper_noun query")
        else:
            rewritten = self._rewrite_query(query, trace=trace)
            if rewritten != query:
                trace.rewritten_query = rewritten

        bundle = self.retrieve_flow.retrieve(
            rewritten,
            limit=limit,
            trace=trace,
            query_type=query_type,
        )

        if not bundle.hits:
            self._append_warning(trace, "empty retrieval")
            trace.total_latency_ms = sum(stage.duration_ms for stage in trace.stages)
            return GeneratedAnswer(
                query=query,
                answer="No relevant context found in your knowledge base.",
                grounded=False,
                abstained=True,
                trace=trace,
            )

        # Step 2: build context block for prompt
        context_hits = self._select_context_hits(bundle)
        context = self._build_context(context_hits)
        trace.context_node_ids = [hit.node_id for hit in context_hits]

        # Step 2b: fallback — if rewritten query yielded no usable context, retry with raw query
        if not context.strip() and rewritten != query:
            self._append_warning(trace, "rewritten retrieval produced empty context; retrying raw query")
            bundle = self.retrieve_flow.retrieve(query, limit=limit, trace=trace, query_type=query_type)
            context_hits = self._select_context_hits(bundle)
            context = self._build_context(context_hits)
            trace.context_node_ids = [hit.node_id for hit in context_hits]

        if not context.strip():
            self._append_warning(trace, "empty retrieval context")
            trace.total_latency_ms = sum(stage.duration_ms for stage in trace.stages)
            return GeneratedAnswer(
                query=query,
                answer="No relevant context found in your knowledge base.",
                grounded=False,
                abstained=True,
                trace=trace,
            )

        # Step 3: call LLM with strict grounding instructions
        messages = [
            ChatMessage(role="system", content=self.SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        llm_started = perf_counter()
        response = self.llm.chat(messages)
        trace.stages.append(
            StageTrace(
                name="llm_generate",
                duration_ms=(perf_counter() - llm_started) * 1000.0,
                hit_count=len(trace.context_node_ids),
            )
        )
        self._accumulate_usage(trace, response.raw)

        # Step 4: extract citations from response
        answer_text = response.text
        abstained_text = self._abstained_response(answer_text, bundle)
        if abstained_text is not None:
            self._append_warning(trace, "llm abstained")
            trace.total_latency_ms = sum(stage.duration_ms for stage in trace.stages)
            return GeneratedAnswer(
                query=query,
                answer=abstained_text,
                citations=[],
                grounded=False,
                abstained=True,
                trace=trace,
            )

        citations = self._extract_citations(answer_text, bundle)

        # Step 4b: fallback — if LLM produced no citations, inject top source hit
        if not citations:
            top = next((h for h in bundle.hits if h.node_id.startswith("src:")), bundle.hits[0])
            answer_text = answer_text + f"\n\n[[{top.node_id}]]"
            citations = [Citation(source_id=top.node_id, title=top.title, excerpt=top.excerpt[:150])]

        # Step 5: verify grounding
        hit_ids = {hit.node_id for hit in bundle.hits}
        hit_source_ids = {self._resolve_source_id(hit.node_id) for hit in bundle.hits}
        grounded = len(citations) > 0 and all(
            citation.source_id in hit_ids or citation.source_id in hit_source_ids
            for citation in citations
        )
        trace.total_latency_ms = sum(stage.duration_ms for stage in trace.stages)

        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            grounded=grounded,
            abstained=False,
            trace=trace,
        )

    def _build_context(self, hits: list) -> str:
        """Format retrieved hits into a context block for the prompt."""
        lines = []
        for hit in hits:
            lines.append(f"[{hit.node_id}] {hit.title}")
            lines.append(hit.excerpt)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _select_context_hits(bundle: RetrievalBundle) -> list:
        return [
            hit
            for hit in bundle.hits
            if hit.node_id.startswith("src:") or hit.score > 0.55
        ]

    # Matches the numeric chunk suffix, e.g. ":0009" at the end of an ID
    _CHUNK_SUFFIX = re.compile(r":\d+$")
    _REWRITE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.-]*")
    _REWRITE_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "s", "the", "to", "what",
        "which", "with",
    }

    def _extract_citations(
        self, answer_text: str, bundle: RetrievalBundle
    ) -> list[Citation]:
        """Find [[src:...]] or [[chk:...]] references in the answer.

        Chunk IDs (ending in :NNNN) are resolved to their parent src: node
        whenever possible. If the source node is not part of the returned hit
        bundle, fall back to the graph store so chunk-heavy bundles still
        produce source-level citations.
        """
        found_ids = self._CITATION_PATTERN.findall(answer_text)

        hit_map = {hit.node_id: hit for hit in bundle.hits}
        citations = []
        seen: set[str] = set()

        for cited_id in found_ids:
            if cited_id in seen:
                continue
            seen.add(cited_id)

            hit = hit_map.get(cited_id)

            # Always resolve chunk IDs to their parent source node.
            if self._CHUNK_SUFFIX.search(cited_id):
                parent_id = self._resolve_source_id(cited_id)
                parent_hit = hit_map.get(parent_id)
                if parent_hit is not None:
                    hit = parent_hit
                else:
                    parent_node = self.retrieve_flow.graph_store.get_node(parent_id)
                    if parent_node is not None:
                        citations.append(Citation(
                            source_id=parent_id,
                            title=parent_node.title,
                            excerpt=(parent_node.text or "")[:150],
                        ))
                        continue

            if hit:
                citations.append(Citation(
                    source_id=hit.node_id,
                    title=hit.title,
                    excerpt=hit.excerpt[:150],
                ))

        return citations

    @classmethod
    def _contains_abstention_signal(cls, answer_text: str) -> bool:
        normalized = answer_text.casefold()
        return any(signal in normalized for signal in cls._ABSTENTION_SIGNALS)

    @classmethod
    def _abstained_response(
        cls,
        answer_text: str,
        bundle: RetrievalBundle,
    ) -> str | None:
        has_hedge = cls._contains_abstention_signal(answer_text)
        has_citation = bool(cls._CITATION_PATTERN.search(answer_text))
        max_retrieval_score = max((hit.score for hit in bundle.hits), default=0.0)

        if has_hedge and not has_citation:
            return cls._strip_citation_markers(answer_text)
        if has_hedge and has_citation and max_retrieval_score < cls._ABSTENTION_SCORE_THRESHOLD:
            return cls._strip_citation_markers(answer_text)
        return None

    @classmethod
    def _strip_citation_markers(cls, answer_text: str) -> str:
        cleaned = cls._CITATION_PATTERN.sub("", answer_text)
        return cleaned.strip()

    def _rewrite_query(self, query: str, trace: QueryTrace | None = None) -> str:
        """Use the LLM to suggest a few additional retrieval terms."""
        started_at = perf_counter() if trace is not None else None
        try:
            messages = [
                ChatMessage(
                    role="user",
                    content=self.build_rewrite_prompt(query),
                )
            ]
            response = self.llm.chat(messages)
            if trace is not None and started_at is not None:
                trace.stages.append(
                    StageTrace(
                        name="llm_rewrite",
                        duration_ms=(perf_counter() - started_at) * 1000.0,
                        hit_count=0,
                    )
                )
                self._accumulate_usage(trace, response.raw)
            return self.sanitize_rewritten_query(query, response.text)
        except Exception as exc:
            if trace is not None and started_at is not None:
                trace.stages.append(
                    StageTrace(
                        name="llm_rewrite",
                        duration_ms=(perf_counter() - started_at) * 1000.0,
                        hit_count=0,
                        notes="fallback_to_original_query",
                    )
                )
                self._append_warning(trace, f"query rewrite failed: {type(exc).__name__}: {exc}")
            print(f"[_rewrite_query] LLM call failed: {exc!r} — falling back to original query")
            return query

    @classmethod
    def sanitize_rewritten_query(cls, original_query: str, rewritten_query: str) -> str:
        """Append a small number of safe retrieval terms to the original query."""

        llm_tokens = cls._extract_rewrite_tokens(rewritten_query)
        if not llm_tokens:
            return original_query

        extra_tokens: list[str] = []
        for token in llm_tokens:
            normalized = token.casefold()
            if normalized in cls._REWRITE_STOPWORDS:
                continue
            if cls._is_suspicious_rewrite_token(token):
                return original_query
            extra_tokens.append(token)
            if len(extra_tokens) >= 8:
                break

        if not extra_tokens:
            return original_query
        return f"{original_query} {' '.join(extra_tokens)}"

    @classmethod
    def build_rewrite_prompt(cls, query: str) -> str:
        return cls._REWRITE_PROMPT_TEMPLATE.format(question=query)

    @classmethod
    def _resolve_source_id(cls, node_id: str) -> str:
        if node_id.startswith("src:"):
            return node_id
        if cls._CHUNK_SUFFIX.search(node_id):
            base = cls._CHUNK_SUFFIX.sub("", node_id)
            return "src:" + base.split(":", 1)[1]
        return node_id

    @classmethod
    def _extract_rewrite_tokens(cls, rewritten_query: str) -> list[str]:
        return cls._REWRITE_TOKEN.findall(rewritten_query)

    @classmethod
    def _is_suspicious_rewrite_token(cls, token: str) -> bool:
        uppercase_after_first = sum(1 for char in token[1:] if char.isupper())
        return len(token) > 24 or uppercase_after_first > 4

    @staticmethod
    def _accumulate_usage(trace: QueryTrace, raw: dict) -> None:
        prompt_tokens = 0
        completion_tokens = 0

        usage = raw.get("usage")
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )

        prompt_tokens += int(raw.get("prompt_tokens", raw.get("prompt_eval_count", 0)) or 0)
        completion_tokens += int(raw.get("completion_tokens", raw.get("eval_count", 0)) or 0)

        trace.prompt_tokens += prompt_tokens
        trace.completion_tokens += completion_tokens

    @staticmethod
    def _append_warning(trace: QueryTrace, warning: str) -> None:
        if warning not in trace.warnings:
            trace.warnings.append(warning)
