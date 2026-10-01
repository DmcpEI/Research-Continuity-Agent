"""Generation flow — grounded answers with citation enforcement."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from time import perf_counter

from pydantic import BaseModel, Field

from rca.config.settings import Settings, get_settings
from rca.contracts.trace import QueryTrace, StageTrace
from rca.flows.retrieve_flow import RetrievalBundle, RetrieveFlow
from rca.llm.client import ChatMessage, EchoLLMClient, LLMClient
from rca.llm.factory import get_llm_client
from rca.retrieval.query_classifier import QueryType, classify_query

logger = logging.getLogger(__name__)


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
    failure_labels: list[str] = Field(default_factory=list)
    trace: QueryTrace | None = None


class GroundingCheck(BaseModel):
    valid: bool = False
    failure_labels: list[str] = Field(default_factory=list)


class AbstentionDecision(BaseModel):
    abstain: bool = False
    answer: str | None = None
    failure_labels: list[str] = Field(default_factory=list)


class GenerateFlow:
    """Generate grounded answers from retrieved context."""

    SYSTEM_PROMPT = """You are a research assistant with access to a personal knowledge base.
Your job is to answer questions using ONLY the context provided below.
Rules:
- If the context contains enough information to answer, every factual claim MUST be followed immediately by [[source_id]] where source_id is the EXACT ID shown in square brackets at the start of the relevant context block
- Use the full ID as written, e.g. [[src:pdf/paper_name]] or [[chk:pdf/paper_name:0012]]
- If the context does not contain enough information, say so clearly and do not include any [[...]] citations
- Never invent facts, authors, or results not present in the context
- Use clean markdown formatting with readable structure.
- Default style: concise prose paragraph(s), not a list.
- Use bullet points only when:
    1) the user explicitly asks for points/list/steps, or
    2) the answer naturally contains multiple distinct items that are clearer as a list.
- If the user asks for more detail, expand with specific evidence from context and keep structure readable.
- Use blank lines between sections when formatting longer answers.
- Keep tone precise and informative — this is a research tool, not a chatbot
- If no retrieved chunks directly support your answer with specific facts, you must abstain
- Do not generate answers that only restate the question without adding factual content from the sources"""
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
    # Calibrated on eval dev split for cross-encoder/ms-marco-MiniLM-L-6-v2 logits
    # (eval/calibrate_abstention.py); recalibrate if the reranker model changes.
    _RERANK_ABSTENTION_THRESHOLD = -3.618
    _CONTENT_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.-]*")
    _LOOSE_CITATION_PATTERN = re.compile(r"(?<!\[)\[((?:src|chk):[^\]\s]+)\](?!\])")
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
        rewrite_llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieve_flow = retrieve_flow or RetrieveFlow()

        if llm_client is not None:
            self.llm = llm_client
        elif self.settings.generation_model:
            self.llm = get_llm_client(self.settings)
        else:
            self.llm = EchoLLMClient()

        # Optional dedicated model/client for query rewriting.
        # If omitted, rewriting uses the same client as generation.
        self.rewrite_llm = rewrite_llm or self.llm

    def generate_answer(
        self,
        query: str,
        limit: int = 5,
        trace: QueryTrace | None = None,
        conversation_messages: Sequence[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        trace = trace or QueryTrace(query=query)
        trace.model = getattr(self.llm, "model", self.llm.__class__.__name__)
        query_type = classify_query(query)
        trace.query_type = query_type.value

        if query_type is QueryType.conversational:
            messages = [
                ChatMessage(role="system", content=self.SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        f"{self._build_conversation_history_block(conversation_messages)}"
                        f"Question: {query}"
                    ),
                ),
            ]
            llm_started = perf_counter()
            response = self.llm.chat(messages)
            trace.stages.append(
                StageTrace(
                    name="llm_generate",
                    duration_ms=(perf_counter() - llm_started) * 1000.0,
                    hit_count=0,
                )
            )
            self._accumulate_usage(trace, response.raw)
            trace.total_latency_ms = sum(stage.duration_ms for stage in trace.stages)
            return GeneratedAnswer(
                query=query,
                answer=response.text.strip() or "I don't have prior context to answer that yet.",
                citations=[],
                grounded=False,
                abstained=False,
                trace=trace,
            )

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
                failure_labels=["empty_retrieval", "low_retrieval_confidence"],
                trace=trace,
            )

        # Step 2: build context block for prompt
        context_hits = self._select_context_hits(bundle)
        context = self._build_context(context_hits)
        trace.context_node_ids = [hit.node_id for hit in context_hits]

        # Step 2b: fallback — if rewritten query yielded no usable context, retry with raw query
        if not context.strip() and rewritten != query:
            self._append_warning(
                trace, "rewritten retrieval produced empty context; retrying raw query"
            )
            bundle = self.retrieve_flow.retrieve(
                query, limit=limit, trace=trace, query_type=query_type
            )
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
                failure_labels=["empty_context", "low_retrieval_confidence"],
                trace=trace,
            )

        # Step 2c: retrieval gate — skip generation when the reranker finds no strong match
        rerank_scores = [
            hit.metadata["rerank_score"] for hit in bundle.hits if "rerank_score" in hit.metadata
        ]
        if rerank_scores and max(rerank_scores) <= self._RERANK_ABSTENTION_THRESHOLD:
            self._append_warning(trace, f"rerank gate: max rerank score {max(rerank_scores):.3f}")
            trace.total_latency_ms = sum(stage.duration_ms for stage in trace.stages)
            return GeneratedAnswer(
                query=query,
                answer="I couldn't find evidence for this in your knowledge base.",
                grounded=False,
                abstained=True,
                failure_labels=["low_rerank_score", "low_retrieval_confidence"],
                trace=trace,
            )

        # Step 3: call LLM with strict grounding instructions
        messages = [
            ChatMessage(role="system", content=self.SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"{self._build_conversation_history_block(conversation_messages)}"
                    f"Context:\n{context}\n\nQuestion: {query}"
                ),
            ),
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
        answer_text = self._normalize_loose_citation_markers(response.text, bundle)
        citations = self._extract_citations(answer_text, bundle)
        grounding = self._verify_grounding(answer_text, citations, bundle)
        abstention = self._decide_abstention(answer_text, bundle, grounding)
        self._append_failure_warnings(trace, abstention.failure_labels)

        if abstention.abstain:
            trace.total_latency_ms = sum(stage.duration_ms for stage in trace.stages)
            return GeneratedAnswer(
                query=query,
                answer=abstention.answer or self._strip_citation_markers(answer_text),
                citations=[],
                grounded=False,
                abstained=True,
                failure_labels=abstention.failure_labels,
                trace=trace,
            )

        grounded = grounding.valid
        trace.total_latency_ms = sum(stage.duration_ms for stage in trace.stages)

        if not self._strip_citation_markers(answer_text) and bundle.hits:
            logger.warning("Empty model answer for query %r (raw=%r)", query, answer_text)
            top = citations[0] if citations else None
            title = top.title if top else bundle.hits[0].title
            source_id = top.source_id if top else self._resolve_source_id(bundle.hits[0].node_id)
            if query_type is QueryType.proper_noun:
                if title:
                    answer_text = (
                        f"I found **{title}** (`{source_id}`) in your knowledge base, "
                        "but couldn't generate a summary. "
                        "Try asking a more specific question about it."
                    )
                else:
                    answer_text = (
                        f"I found a source (`{source_id}`) in your knowledge base, "
                        "but couldn't generate a summary. "
                        "Try asking a more specific question about it."
                    )
            else:
                answer_text = (
                    "I found relevant sources but the response was empty. "
                    "Please try rephrasing your question."
                )

        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            grounded=grounded,
            abstained=False,
            failure_labels=grounding.failure_labels,
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
    def _build_conversation_history_block(
        conversation_messages: Sequence[dict[str, str]] | None,
    ) -> str:
        if not conversation_messages:
            return ""

        lines = ["Conversation history:"]
        for item in conversation_messages:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue

            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"- {speaker}: {content}")

        if len(lines) == 1:
            return ""
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _select_context_hits(bundle: RetrievalBundle) -> list:
        return [hit for hit in bundle.hits if hit.node_id.startswith("src:") or hit.score > 0.55]

    # Matches the numeric chunk suffix, e.g. ":0009" at the end of an ID
    _CHUNK_SUFFIX = re.compile(r":\d+$")
    _REWRITE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.-]*")
    _REWRITE_STOPWORDS = {
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
        "s",
        "the",
        "to",
        "what",
        "which",
        "with",
    }

    def _extract_citations(self, answer_text: str, bundle: RetrievalBundle) -> list[Citation]:
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
                    graph_store = getattr(self.retrieve_flow, "graph_store", None)
                    parent_node = graph_store.get_node(parent_id) if graph_store else None
                    if parent_node is not None:
                        citations.append(
                            Citation(
                                source_id=parent_id,
                                title=parent_node.title,
                                excerpt=(parent_node.text or "")[:150],
                            )
                        )
                        continue

            if hit:
                citations.append(
                    Citation(
                        source_id=hit.node_id,
                        title=hit.title,
                        excerpt=hit.excerpt[:150],
                    )
                )

        return citations

    def _normalize_loose_citation_markers(self, answer_text: str, bundle: RetrievalBundle) -> str:
        """Normalize model-emitted [src:...] markers when they point at retrieved evidence."""

        hit_ids = {hit.node_id for hit in bundle.hits}
        hit_source_ids = {self._resolve_source_id(hit.node_id) for hit in bundle.hits}

        def replace(match: re.Match[str]) -> str:
            cited_id = match.group(1).strip()
            cited_source_id = self._resolve_source_id(cited_id)
            if cited_id in hit_ids or cited_source_id in hit_source_ids:
                return f"[[{cited_id}]]"
            return match.group(0)

        return self._LOOSE_CITATION_PATTERN.sub(replace, answer_text)

    @classmethod
    def _contains_abstention_signal(cls, answer_text: str) -> bool:
        normalized = answer_text.casefold()
        return any(signal in normalized for signal in cls._ABSTENTION_SIGNALS)

    def _verify_grounding(
        self,
        answer_text: str,
        citations: list[Citation],
        bundle: RetrievalBundle,
    ) -> GroundingCheck:
        labels: list[str] = []
        raw_citation_ids = self._CITATION_PATTERN.findall(answer_text)
        hit_ids = {hit.node_id for hit in bundle.hits}
        hit_source_ids = {self._resolve_source_id(hit.node_id) for hit in bundle.hits}

        if not raw_citation_ids:
            return GroundingCheck(valid=False, failure_labels=["missing_citation"])
        if not citations:
            return GroundingCheck(valid=False, failure_labels=["invalid_citation"])

        for citation in citations:
            if citation.source_id not in hit_ids and citation.source_id not in hit_source_ids:
                labels.append("invalid_citation")
                break

        if not labels and not self._answer_supported_by_cited_evidence(
            answer_text, citations, bundle
        ):
            labels.append("unsupported_claim")

        return GroundingCheck(valid=not labels, failure_labels=self._dedupe_labels(labels))

    def _decide_abstention(
        self,
        answer_text: str,
        bundle: RetrievalBundle,
        grounding: GroundingCheck,
    ) -> AbstentionDecision:
        labels = list(grounding.failure_labels)
        has_hedge = self._contains_abstention_signal(answer_text)
        max_retrieval_score = max((hit.score for hit in bundle.hits), default=0.0)

        if has_hedge:
            labels.append("llm_abstained")
        if max_retrieval_score < self._ABSTENTION_SCORE_THRESHOLD:
            labels.append("low_retrieval_confidence")

        labels = self._dedupe_labels(labels)
        should_abstain = False
        if has_hedge and (
            not grounding.valid or max_retrieval_score < self._ABSTENTION_SCORE_THRESHOLD
        ):
            should_abstain = True
        if "invalid_citation" in labels or "unsupported_claim" in labels:
            should_abstain = True
        if "missing_citation" in labels and "low_retrieval_confidence" in labels:
            should_abstain = True

        return AbstentionDecision(
            abstain=should_abstain,
            answer=self._strip_citation_markers(answer_text) if should_abstain else None,
            failure_labels=labels,
        )

    def _answer_supported_by_cited_evidence(
        self,
        answer_text: str,
        citations: list[Citation],
        bundle: RetrievalBundle,
    ) -> bool:
        answer_tokens = self._content_tokens(self._strip_citation_markers(answer_text))
        if not answer_tokens:
            return False

        evidence_by_source: dict[str, list[str]] = {}
        for hit in bundle.hits:
            source_id = self._resolve_source_id(hit.node_id)
            evidence_by_source.setdefault(source_id, []).extend([hit.title, hit.excerpt])
            evidence_by_source.setdefault(hit.node_id, []).extend([hit.title, hit.excerpt])

        for citation in citations:
            evidence_text = " ".join(evidence_by_source.get(citation.source_id, []))
            if not evidence_text:
                continue
            if self._has_basic_support(answer_tokens, evidence_text):
                return True
        return False

    @classmethod
    def _has_basic_support(cls, answer_tokens: set[str], evidence_text: str) -> bool:
        evidence_tokens = cls._content_tokens(evidence_text)
        shared = answer_tokens & evidence_tokens
        if len(shared) >= 2:
            return True
        if len(answer_tokens) <= 4 and any(len(token) >= 4 for token in shared):
            return True
        return False

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        stopwords = cls._REWRITE_STOPWORDS | {
            "about",
            "above",
            "below",
            "but",
            "context",
            "describe",
            "describes",
            "did",
            "do",
            "given",
            "into",
            "not",
            "provided",
            "requested",
            "that",
            "this",
            "was",
            "were",
        }
        return {
            cleaned
            for token in cls._CONTENT_TOKEN.findall(text)
            if len(cleaned := token.strip("._-+").casefold()) >= 3 and cleaned not in stopwords
        }

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
            response = self.rewrite_llm.chat(messages)
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

    @classmethod
    def _append_failure_warnings(cls, trace: QueryTrace, labels: list[str]) -> None:
        for label in labels:
            cls._append_warning(trace, f"failure:{label}")

    @staticmethod
    def _dedupe_labels(labels: list[str]) -> list[str]:
        deduped: list[str] = []
        for label in labels:
            if label not in deduped:
                deduped.append(label)
        return deduped
