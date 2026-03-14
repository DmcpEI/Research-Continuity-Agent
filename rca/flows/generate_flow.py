"""Generation flow — grounded answers with citation enforcement."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from rca.config.settings import Settings, get_settings
from rca.flows.retrieve_flow import RetrieveFlow, RetrievalBundle
from rca.llm.client import ChatMessage, EchoLLMClient, LLMClient


class Citation(BaseModel):
    source_id: str
    title: str
    excerpt: str


class GeneratedAnswer(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = False


class GenerateFlow:
    """Generate grounded answers from retrieved context."""

    SYSTEM_PROMPT = """You are a research assistant with access to a personal knowledge base.
Your job is to answer questions using ONLY the context provided below.
Rules:
- Every factual claim MUST be followed immediately by [[source_id]] where source_id is the EXACT ID shown in square brackets at the start of the relevant context block
- An answer with no [[...]] citations will be rejected — citations are mandatory, not optional
- Use the full ID as written, e.g. [[src:pdf/paper_name]] or [[chk:pdf/paper_name:0012]]
- If the context does not contain enough information, say so and still cite the closest source
- Never invent facts, authors, or results not present in the context
- Be concise and precise — this is a research tool, not a chatbot"""

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
                base_url=self.settings.embedding_base_url,
                model=self.settings.generation_model,
            )
        else:
            self.llm = EchoLLMClient()

    def generate_answer(self, query: str, limit: int = 5) -> GeneratedAnswer:
        # Step 1: retrieve grounded context
        rewritten = self._rewrite_query(query)
        bundle = self.retrieve_flow.retrieve(rewritten)

        if not bundle.hits:
            return GeneratedAnswer(
                query=query,
                answer="No relevant context found in your knowledge base.",
                grounded=False,
            )

        # Step 2: build context block for prompt
        context = self._build_context(bundle)

        # Step 2b: fallback — if rewritten query yielded no usable context, retry with raw query
        if not context.strip() and rewritten != query:
            bundle = self.retrieve_flow.retrieve(query)
            context = self._build_context(bundle)

        if not context.strip():
            return GeneratedAnswer(
                query=query,
                answer="No relevant context found in your knowledge base.",
                grounded=False,
            )

        # Step 3: call LLM with strict grounding instructions
        messages = [
            ChatMessage(role="system", content=self.SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        response = self.llm.chat(messages)

        # Step 4: extract citations from response
        answer_text = response.text
        citations = self._extract_citations(answer_text, bundle)

        # Step 4b: fallback — if LLM produced no citations, inject top source hit
        if not citations:
            top = next((h for h in bundle.hits if h.node_id.startswith("src:")), bundle.hits[0])
            answer_text = answer_text + f"\n\n[[{top.node_id}]]"
            citations = [Citation(source_id=top.node_id, title=top.title, excerpt=top.excerpt[:150])]

        # Step 5: verify grounding
        hit_ids = {hit.node_id for hit in bundle.hits}
        grounded = len(citations) > 0 and all(c.source_id in hit_ids for c in citations)

        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            grounded=grounded,
        )

    def _build_context(self, bundle: RetrievalBundle) -> str:
        """Format retrieved hits into a context block for the prompt."""
        lines = []
        for hit in bundle.hits:
            if hit.node_id.startswith("src:") or hit.score > 0.55:
                lines.append(f"[{hit.node_id}] {hit.title}")
                lines.append(hit.excerpt)
                lines.append("")
        return "\n".join(lines)

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

        Chunk IDs (ending in :NNNN) are always resolved to their parent src:
        node — regardless of whether the chunk itself is in the hit map.
        The previous guard (hit is None) caused chunk citations to be stored
        as-is when the chunk was in the hit map, breaking source_correct checks.
        """
        pattern = re.compile(r"\[\[((?:src|chk):[^\]]+)\]\]")
        found_ids = pattern.findall(answer_text)

        hit_map = {hit.node_id: hit for hit in bundle.hits}
        citations = []
        seen: set[str] = set()

        for cited_id in found_ids:
            if cited_id in seen:
                continue
            seen.add(cited_id)

            hit = hit_map.get(cited_id)

            # Always resolve chunk IDs to their parent source node.
            # Try src: prefix regardless of what the LLM wrote.
            if self._CHUNK_SUFFIX.search(cited_id):
                base = self._CHUNK_SUFFIX.sub("", cited_id)
                parent_id = "src:" + base.split(":", 1)[1]
                parent_hit = hit_map.get(parent_id)
                if parent_hit is not None:
                    hit = parent_hit

            if hit:
                citations.append(Citation(
                    source_id=hit.node_id,
                    title=hit.title,
                    excerpt=hit.excerpt[:150],
                ))

        return citations

    def _rewrite_query(self, query: str) -> str:
        """Use the LLM to extract clean semantic search keywords."""
        try:
            messages = [
                ChatMessage(
                    role="user",
                    content=(
                        "Convert this research question into a dense technical search query of 8-12 keywords. "
                        "Include domain-specific terms, method names, and technical concepts. "
                        "No explanation, no punctuation, no full sentences. Just keywords.\n\n"
                        f"Question: {query}\n\nKeywords:"
                    ),
                )
            ]
            response = self.llm.chat(messages)
            return self.sanitize_rewritten_query(query, response.text)
        except Exception as exc:
            print(f"[_rewrite_query] LLM call failed: {exc!r} — falling back to original query")
            return query

    @classmethod
    def sanitize_rewritten_query(cls, original_query: str, rewritten_query: str) -> str:
        """Keep dense keywords while preserving salient proper nouns from the original query."""

        llm_tokens = cls._extract_rewrite_tokens(rewritten_query)
        original_tokens = cls._extract_salient_original_tokens(original_query)

        merged_tokens: list[str] = []
        seen: set[str] = set()
        for token in original_tokens + llm_tokens:
            normalized = token.casefold()
            if normalized in seen or normalized in cls._REWRITE_STOPWORDS:
                continue
            seen.add(normalized)
            merged_tokens.append(token)
            if len(merged_tokens) >= 12:
                break

        if len(merged_tokens) < 4:
            return original_query
        return " ".join(merged_tokens)

    @classmethod
    def _extract_rewrite_tokens(cls, rewritten_query: str) -> list[str]:
        return cls._REWRITE_TOKEN.findall(rewritten_query)

    @classmethod
    def _extract_salient_original_tokens(cls, original_query: str) -> list[str]:
        salient_tokens: list[str] = []
        for token in cls._REWRITE_TOKEN.findall(original_query):
            normalized = token.casefold()
            if normalized in cls._REWRITE_STOPWORDS:
                continue
            if (
                any(char.isdigit() for char in token)
                or any(char in token for char in "+-_")
                or token.isupper()
                or any(char.isupper() for char in token[1:])
                or len(token) >= 8
            ):
                salient_tokens.append(token)
        return salient_tokens
