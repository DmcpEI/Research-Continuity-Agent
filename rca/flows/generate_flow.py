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
- Every factual claim must reference a source using [[src:source_id]] notation
- Replace 'source_id' with the actual ID shown in brackets at the start of each context block
- If the context does not contain enough information, say so explicitly
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
        bundle = self.retrieve_flow.retrieve(query, limit=limit)

        if not bundle.hits:
            return GeneratedAnswer(
                query=query,
                answer="No relevant context found in your knowledge base.",
                grounded=True,
            )

        # Step 2: build context block for prompt
        context = self._build_context(bundle)

        # Step 3: call LLM with strict grounding instructions
        messages = [
            ChatMessage(role="system", content=self.SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        response = self.llm.chat(messages)

        # Step 4: extract citations from response
        citations = self._extract_citations(response.text, bundle)

        # Step 5: verify grounding
        hit_ids = {hit.node_id for hit in bundle.hits}
        grounded = len(citations) > 0 and all(c.source_id in hit_ids for c in citations)

        return GeneratedAnswer(
            query=query,
            answer=response.text,
            citations=citations,
            grounded=grounded,
        )

    def _build_context(self, bundle: RetrievalBundle) -> str:
        """Format retrieved hits into a context block for the prompt."""
        lines = []
        for hit in bundle.hits:
            if hit.node_id.startswith("src:") or hit.score > 0.7:
                lines.append(f"[{hit.node_id}] {hit.title}")
                lines.append(hit.excerpt)
                lines.append("")
        return "\n".join(lines)

    def _extract_citations(
        self, answer_text: str, bundle: RetrievalBundle
    ) -> list[Citation]:
        """Find [[src:...]] or [[chk:...]] references in the answer."""
        pattern = re.compile(r"\[\[((?:src|chk):[^\]]+)\]\]")
        found_ids = pattern.findall(answer_text)

        hit_map = {hit.node_id: hit for hit in bundle.hits}
        citations = []
        seen: set[str] = set()

        for source_id in found_ids:
            if source_id in seen:
                continue
            seen.add(source_id)
            hit = hit_map.get(source_id)
            if hit:
                citations.append(Citation(
                    source_id=source_id,
                    title=hit.title,
                    excerpt=hit.excerpt[:150],
                ))

        return citations