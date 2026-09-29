from __future__ import annotations

import pytest

from rca.contracts.nodes import Node, NodeKind
from rca.flows.generate_flow import GenerateFlow
from rca.flows.retrieve_flow import RetrievalBundle, RetrievalHit
from rca.llm.client import ChatResponse, LLMClient
from rca.retrieval.query_classifier import QueryType


class StubLLMClient(LLMClient):
    def __init__(self, responses: list[str] | None = None) -> None:
        self.model = "stub-model"
        self.calls = 0
        self.last_messages = None
        self.responses = responses or [
            "JamPacker components planning recovery packing efficiency",
            "JamPacker combines planning and recovery modules. [[src:pdf/jampacker]]",
        ]

    def chat(self, messages):
        self.calls += 1
        self.last_messages = messages
        index = min(self.calls - 1, len(self.responses) - 1)
        return ChatResponse(
            text=self.responses[index],
            raw={
                "prompt_eval_count": 11 if self.calls == 1 else 29,
                "eval_count": 4 if self.calls == 1 else 8,
            },
        )

    def embed(self, texts: list[str], dimensions: int = 32) -> list[list[float]]:
        raise NotImplementedError


class StubRetrieveFlow:
    def __init__(self, score: float = 0.92) -> None:
        self.score = score
        self.queries: list[str] = []
        self.query_types: list[QueryType | None] = []
        self.graph_store = None

    def retrieve(
        self, query: str, limit: int = 10, trace=None, query_type: QueryType | None = None
    ) -> RetrievalBundle:
        self.queries.append(query)
        self.query_types.append(query_type)
        return RetrievalBundle(
            query=query,
            hits=[
                RetrievalHit(
                    node_id="src:pdf/jampacker",
                    score=self.score,
                    title="JamPacker",
                    excerpt=(
                        "JamPacker uses planning and recovery modules for robotic bin packing "
                        "and placement heuristics."
                    ),
                    metadata={},
                )
            ],
            related_edges=[],
            trace=trace,
        )


def test_generate_answer_conversational_skips_retrieval() -> None:
    class ExplodingRetrieveFlow(StubRetrieveFlow):
        def retrieve(
            self, query: str, limit: int = 10, trace=None, query_type: QueryType | None = None
        ) -> RetrievalBundle:
            raise AssertionError("retrieval should be skipped for conversational queries")

    llm = StubLLMClient(responses=["Yes, I can see our prior messages."])
    flow = GenerateFlow(retrieve_flow=ExplodingRetrieveFlow(), llm_client=llm)

    result = flow.generate_answer(
        "Why didn't you respond?",
        conversation_messages=[
            {"role": "user", "content": "Is there a paper about LLM-Pack?"},
            {"role": "assistant", "content": "Yes, I found it."},
        ],
    )

    assert result.abstained is False
    assert result.grounded is False
    assert result.citations == []
    assert "prior messages" in result.answer


def test_generate_answer_empty_response_falls_back_to_title() -> None:
    flow = GenerateFlow(retrieve_flow=StubRetrieveFlow(), llm_client=StubLLMClient(responses=[""]))

    result = flow.generate_answer("Is JamPacker in the database?")

    assert result.abstained is False
    assert result.grounded is False
    assert "missing_citation" in result.failure_labels
    assert "JamPacker" in result.answer
    assert "src:pdf/jampacker" in result.answer
    assert "couldn't generate a summary" in result.answer


def test_generate_answer_empty_response_generic_fallback_for_conceptual_query() -> None:
    flow = GenerateFlow(retrieve_flow=StubRetrieveFlow(), llm_client=StubLLMClient(responses=[""]))

    result = flow.generate_answer("What approaches exist for robotic bin packing?")

    assert result.abstained is False
    assert result.grounded is False
    assert "missing_citation" in result.failure_labels
    assert "relevant sources" in result.answer


def test_generate_answer_attaches_query_trace() -> None:
    flow = GenerateFlow(retrieve_flow=StubRetrieveFlow(), llm_client=StubLLMClient())

    result = flow.generate_answer("what approaches exist for robotic bin packing?")

    assert result.abstained is False
    assert result.trace is not None
    assert result.trace.model == "stub-model"
    assert result.trace.query_type == "conceptual"
    assert result.trace.rewritten_query is not None
    assert result.trace.context_node_ids == ["src:pdf/jampacker"]
    assert [stage.name for stage in result.trace.stages] == ["llm_rewrite", "llm_generate"]
    assert result.trace.prompt_tokens == 40
    assert result.trace.completion_tokens == 12
    assert result.trace.total_latency_ms > 0.0


def test_generate_answer_abstains_on_missing_information_signal() -> None:
    flow = GenerateFlow(
        retrieve_flow=StubRetrieveFlow(),
        llm_client=StubLLMClient(
            responses=[
                "JamPacker components planning recovery packing efficiency",
                "The provided context does not contain information about the requested metric.",
            ]
        ),
    )

    result = flow.generate_answer("What metric was not reported?")

    assert result.abstained is True
    assert result.grounded is False
    assert result.citations == []
    assert "[[" not in result.answer
    assert "does not contain information" in result.answer


def test_generate_answer_does_not_abstain_when_hedged_response_includes_citation() -> None:
    flow = GenerateFlow(
        retrieve_flow=StubRetrieveFlow(score=0.92),
        llm_client=StubLLMClient(
            responses=[
                "JamPacker components planning recovery packing efficiency",
                "The provided context does not contain information about the requested metric, but it does describe JamPacker's core modules. [[src:pdf/jampacker]]",
            ]
        ),
    )

    result = flow.generate_answer("What metric was not reported?")

    assert result.abstained is False
    assert result.grounded is True
    assert [citation.source_id for citation in result.citations] == ["src:pdf/jampacker"]
    assert result.answer.endswith("[[src:pdf/jampacker]]")


def test_generate_answer_abstains_on_hedged_citation_when_retrieval_confidence_is_low() -> None:
    flow = GenerateFlow(
        retrieve_flow=StubRetrieveFlow(score=0.40),
        llm_client=StubLLMClient(
            responses=[
                "JamPacker components planning recovery packing efficiency",
                "The provided context does not contain information about the requested metric. [[src:pdf/jampacker]]",
            ]
        ),
    )

    result = flow.generate_answer("What metric was not reported?")

    assert result.abstained is True
    assert result.grounded is False
    assert result.citations == []
    assert "[[" not in result.answer
    assert "does not contain information" in result.answer


def test_generate_answer_marks_uncited_supported_answer_as_ungrounded() -> None:
    flow = GenerateFlow(
        retrieve_flow=StubRetrieveFlow(),
        llm_client=StubLLMClient(
            responses=[
                "JamPacker components planning recovery packing efficiency",
                "JamPacker combines planning and recovery modules.",
            ]
        ),
    )

    result = flow.generate_answer("What are the two main components of JamPacker?")

    assert result.abstained is False
    assert result.grounded is False
    assert result.citations == []
    assert result.failure_labels == ["missing_citation"]
    assert not result.answer.endswith("[[src:pdf/jampacker]]")


def test_generate_answer_normalizes_single_bracket_citations_when_resolvable() -> None:
    flow = GenerateFlow(
        retrieve_flow=StubRetrieveFlow(),
        llm_client=StubLLMClient(
            responses=[
                "JamPacker combines planning and recovery modules. [src:pdf/jampacker]",
            ]
        ),
    )

    result = flow.generate_answer("What are the two main components of JamPacker?")

    assert result.abstained is False
    assert result.grounded is True
    assert result.answer.endswith("[[src:pdf/jampacker]]")
    assert [citation.source_id for citation in result.citations] == ["src:pdf/jampacker"]


def test_generate_answer_skips_rewrite_for_proper_noun_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieve_flow = StubRetrieveFlow()
    flow = GenerateFlow(
        retrieve_flow=retrieve_flow,
        llm_client=StubLLMClient(
            responses=["DBLF is the heuristic used by JamPacker. [[src:pdf/jampacker]]"]
        ),
    )

    def fail_rewrite(query: str, trace=None) -> str:
        raise AssertionError("proper_noun queries should skip rewrite")

    monkeypatch.setattr(flow, "_rewrite_query", fail_rewrite)

    result = flow.generate_answer("What is DBLF in JamPacker?")

    assert result.trace is not None
    assert result.trace.query_type == "proper_noun"
    assert [stage.name for stage in result.trace.stages] == ["llm_generate"]
    assert retrieve_flow.queries == ["What is DBLF in JamPacker?"]
    assert retrieve_flow.query_types == [QueryType.proper_noun]
    assert "rewrite skipped: proper_noun query" in result.trace.warnings


def test_generate_answer_keeps_rewrite_for_conceptual_queries() -> None:
    retrieve_flow = StubRetrieveFlow()
    flow = GenerateFlow(
        retrieve_flow=retrieve_flow,
        llm_client=StubLLMClient(
            responses=[
                "robotic bin packing methods heuristics planning placement stability",
                "Robotic bin packing combines planning and placement heuristics. [[src:pdf/jampacker]]",
            ]
        ),
    )

    result = flow.generate_answer("what approaches exist for robotic bin packing?")

    assert result.trace is not None
    assert result.trace.query_type == "conceptual"
    assert "llm_rewrite" in [stage.name for stage in result.trace.stages]
    assert retrieve_flow.queries == [result.trace.rewritten_query]
    assert retrieve_flow.queries[0] != "what approaches exist for robotic bin packing?"
    assert "robotic" in retrieve_flow.queries[0]
    assert "packing" in retrieve_flow.queries[0]
    assert retrieve_flow.query_types == [QueryType.conceptual]


def test_generate_answer_includes_conversation_history_in_prompt() -> None:
    llm = StubLLMClient(
        responses=[
            "robotic bin packing methods heuristics planning placement stability",
            "JamPacker combines planning and recovery modules. [[src:pdf/jampacker]]",
        ]
    )
    flow = GenerateFlow(retrieve_flow=StubRetrieveFlow(), llm_client=llm)

    flow.generate_answer(
        "Can you be more specific?",
        conversation_messages=[
            {"role": "user", "content": "What approaches exist for robotic bin packing?"},
            {"role": "assistant", "content": "Two approaches are compared in JamPacker."},
        ],
    )

    assert llm.last_messages is not None
    prompt = llm.last_messages[-1].content
    assert "Conversation history:" in prompt
    assert "- User: What approaches exist for robotic bin packing?" in prompt
    assert "- Assistant: Two approaches are compared in JamPacker." in prompt


def test_sanitize_rewritten_query_appends_additional_terms_to_original() -> None:
    original = "Why does the MoMa-LLM paper introduce AUC-E in addition to success and SPL?"

    rewritten = GenerateFlow.sanitize_rewritten_query(
        original,
        "MoMa-LLM AUC-E evaluation metrics",
    )

    assert rewritten == f"{original} MoMa-LLM AUC-E evaluation metrics"


def test_sanitize_rewritten_query_rejects_suspicious_tokens() -> None:
    original = "What two data sources does SGVL train on jointly, and why are they combined?"

    rewritten = GenerateFlow.sanitize_rewritten_query(
        original,
        "SGVL combined rationalespNetGraphVLTensoRFsNeRFmulti-modalintegrationheterogeneousdatafusionmachinelearningcross-domaingeneralization",
    )

    assert rewritten == original


def test_generate_answer_resolves_chunk_citation_to_parent_source_without_source_hit() -> None:
    class StubGraphStore:
        def get_node(self, node_id: str) -> Node | None:
            if node_id == "src:pdf/jampacker":
                return Node(
                    id=node_id,
                    kind=NodeKind.paper,
                    title="JamPacker",
                    text="JamPacker source preview",
                )
            return None

    class ChunkOnlyRetrieveFlow(StubRetrieveFlow):
        def __init__(self) -> None:
            super().__init__()
            self.graph_store = StubGraphStore()

        def retrieve(
            self, query: str, limit: int = 10, trace=None, query_type: QueryType | None = None
        ) -> RetrievalBundle:
            self.queries.append(query)
            self.query_types.append(query_type)
            return RetrievalBundle(
                query=query,
                hits=[
                    RetrievalHit(
                        node_id="chk:pdf/jampacker:0001",
                        score=0.91,
                        title="JamPacker chunk",
                        excerpt="DBLF is described in this chunk.",
                        metadata={"source_id": "src:pdf/jampacker"},
                    )
                ],
                related_edges=[],
                trace=trace,
            )

    flow = GenerateFlow(
        retrieve_flow=ChunkOnlyRetrieveFlow(),
        llm_client=StubLLMClient(
            responses=["DBLF is a JamPacker heuristic. [[chk:pdf/jampacker:0001]]"]
        ),
    )

    result = flow.generate_answer("What is DBLF in JamPacker?")

    assert result.abstained is False
    assert result.grounded is True
    assert [citation.source_id for citation in result.citations] == ["src:pdf/jampacker"]
