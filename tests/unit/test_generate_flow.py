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
    def __init__(self, score: float = 0.92, rerank_score: float | None = None) -> None:
        self.score = score
        self.rerank_score = rerank_score
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
                    metadata={}
                    if self.rerank_score is None
                    else {"rerank_score": self.rerank_score},
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


def test_generate_answer_rerank_gate_abstains_without_llm_generation() -> None:
    llm = StubLLMClient()
    flow = GenerateFlow(
        retrieve_flow=StubRetrieveFlow(rerank_score=-8.0),
        llm_client=llm,
    )

    result = flow.generate_answer("Is JamPacker in the database?")

    assert result.abstained is True
    assert result.grounded is False
    assert result.failure_labels == ["low_rerank_score", "low_retrieval_confidence"]
    assert llm.calls == 0


def test_generate_answer_rerank_gate_passes_strong_match() -> None:
    llm = StubLLMClient(
        responses=["JamPacker combines planning and recovery. [[src:pdf/jampacker]]"]
    )
    flow = GenerateFlow(
        retrieve_flow=StubRetrieveFlow(rerank_score=5.0),
        llm_client=llm,
    )

    result = flow.generate_answer("Is JamPacker in the database?")

    assert result.abstained is False
    assert "low_rerank_score" not in result.failure_labels
    assert llm.calls == 1


def test_generate_answer_prompt_spells_out_citation_tag_and_reminder() -> None:
    llm = StubLLMClient()
    flow = GenerateFlow(retrieve_flow=StubRetrieveFlow(), llm_client=llm)

    flow.generate_answer("How does JamPacker handle recovery?")

    assert llm.last_messages is not None
    prompt = llm.last_messages[-1].content
    assert "[src:pdf/jampacker] JamPacker (cite as [[src:pdf/jampacker]])" in prompt
    # The reminder comes after the question, as the last thing the model reads.
    assert prompt.index("Question: How does JamPacker handle recovery?") < prompt.index(
        GenerateFlow.CITATION_REMINDER
    )
    assert prompt.endswith(GenerateFlow.CITATION_REMINDER)


def test_conversational_prompt_has_no_citation_reminder() -> None:
    llm = StubLLMClient(responses=["Yes, I can see our prior messages."])
    retrieve_flow = StubRetrieveFlow()
    flow = GenerateFlow(retrieve_flow=retrieve_flow, llm_client=llm)

    flow.generate_answer(
        "Why didn't you respond?",
        conversation_messages=[{"role": "user", "content": "Is there a paper about LLM-Pack?"}],
    )

    assert retrieve_flow.queries == []  # conversational path: no context, nothing to cite
    assert llm.last_messages is not None
    assert GenerateFlow.CITATION_REMINDER not in llm.last_messages[-1].content


class LongTextRetrieveFlow(StubRetrieveFlow):
    TAIL = "Its fallback routine re-plans whenever a grasp slips."

    def retrieve(self, query, limit=10, trace=None, query_type=None):
        bundle = super().retrieve(query, limit=limit, trace=trace, query_type=query_type)
        hit = bundle.hits[0]
        full = hit.excerpt + " " + "filler " * 40 + self.TAIL
        bundle.hits[0] = hit.model_copy(update={"text": full, "excerpt": full[:240]})
        return bundle


def test_prompt_gives_the_model_full_hit_text_not_the_short_excerpt() -> None:
    llm = StubLLMClient()
    flow = GenerateFlow(retrieve_flow=LongTextRetrieveFlow(), llm_client=llm)

    flow.generate_answer("How does JamPacker handle recovery?")

    assert llm.last_messages is not None
    assert LongTextRetrieveFlow.TAIL in llm.last_messages[-1].content


def test_grounding_check_accepts_claims_supported_past_the_excerpt_cut() -> None:
    # Cites the hit, but every content word comes from text after the 240-char excerpt.
    llm = StubLLMClient(
        responses=[
            "rewrite",
            "The fallback routine re-plans whenever a grasp slips. [[src:pdf/jampacker]]",
        ]
    )
    flow = GenerateFlow(retrieve_flow=LongTextRetrieveFlow(), llm_client=llm)

    result = flow.generate_answer("What happens when a grasp slips?")

    assert result.grounded is True
    assert "unsupported_claim" not in result.failure_labels


def test_grounding_check_still_flags_a_cited_claim_with_no_support_in_the_text() -> None:
    llm = StubLLMClient(
        # One response for every call, whether or not the query rewrite runs.
        responses=["A quadruped base maps terrain using lidar sweeps. [[src:pdf/jampacker]]"]
    )
    flow = GenerateFlow(retrieve_flow=LongTextRetrieveFlow(), llm_client=llm)

    result = flow.generate_answer("How does JamPacker handle recovery?")

    assert result.grounded is False
    assert "unsupported_claim" in result.failure_labels


_REWRITE = "JamPacker components planning recovery packing efficiency"
_CONCEPTUAL = "what approaches exist for robotic bin packing?"  # rewritten, so two LLM calls


class _NoHitsRetrieveFlow(StubRetrieveFlow):
    def retrieve(
        self, query: str, limit: int = 10, trace=None, query_type: QueryType | None = None
    ) -> RetrievalBundle:
        return RetrievalBundle(query=query, hits=[], related_edges=[], trace=trace)


class _WeakChunkRetrieveFlow(StubRetrieveFlow):
    """Only a low-scoring chunk comes back, so no hit is selected as context."""

    def retrieve(
        self, query: str, limit: int = 10, trace=None, query_type: QueryType | None = None
    ) -> RetrievalBundle:
        hit = RetrievalHit(node_id="chk:pdf/jampacker:0001", score=0.30, title="", excerpt="x")
        return RetrievalBundle(query=query, hits=[hit], related_edges=[], trace=trace)


@pytest.mark.parametrize(
    ("retrieve_flow", "responses", "query", "expected"),
    [
        (StubRetrieveFlow(), ["Yes, I can see it."], "Why didn't you respond?", "conversational"),
        (_NoHitsRetrieveFlow(), [_REWRITE], _CONCEPTUAL, "not_found"),
        (_WeakChunkRetrieveFlow(), [_REWRITE], _CONCEPTUAL, "not_found"),
        (
            StubRetrieveFlow(rerank_score=-8.0),
            [_REWRITE],
            _CONCEPTUAL,
            "not_found",
        ),
        (
            StubRetrieveFlow(score=0.40),
            [_REWRITE, "The provided context does not contain this. [[src:pdf/jampacker]]"],
            _CONCEPTUAL,
            "not_found",
        ),
        (
            StubRetrieveFlow(),
            [_REWRITE, "JamPacker combines planning and recovery modules. [[src:pdf/jampacker]]"],
            _CONCEPTUAL,
            "answered",
        ),
        (
            StubRetrieveFlow(),
            [_REWRITE, "JamPacker combines planning and recovery modules."],
            _CONCEPTUAL,
            "unverified",
        ),
    ],
    ids=[
        "question-about-the-chat",
        "empty-retrieval",
        "empty-context",
        "rerank-gate",
        "post-generation-refusal",
        "cited",
        "uncited",
    ],
)
def test_generate_answer_reports_one_outcome_per_path(
    retrieve_flow, responses, query, expected
) -> None:
    flow = GenerateFlow(retrieve_flow=retrieve_flow, llm_client=StubLLMClient(responses=responses))

    result = flow.generate_answer(query)

    assert result.outcome == expected
    assert result.answer.strip()
    # The legacy flags are views of the outcome and are still serialised for old readers.
    dumped = result.model_dump(mode="json")
    assert dumped["outcome"] == expected
    assert dumped["grounded"] is (expected == "answered")
    assert dumped["abstained"] is (expected == "not_found")


def test_generated_answer_cannot_be_grounded_and_abstained() -> None:
    from pydantic import ValidationError

    from rca.flows.generate_flow import GeneratedAnswer

    with pytest.raises(ValidationError):
        GeneratedAnswer(query="q", answer="a", grounded=True, abstained=True)

    refusal = GeneratedAnswer(query="q", answer="a", outcome="not_found", grounded=True)

    assert refusal.grounded is False
    assert refusal.abstained is True


@pytest.mark.parametrize(
    "reply",
    ["", "   \n", "[[src:pdf/jampacker]]"],
    ids=["empty", "whitespace", "citation-only"],
)
def test_refusal_after_generation_always_has_text(reply) -> None:
    # Low retrieval confidence plus nothing usable from the model: RCA declines, and must say so.
    flow = GenerateFlow(
        retrieve_flow=StubRetrieveFlow(score=0.40),
        llm_client=StubLLMClient(responses=[_REWRITE, reply]),
    )

    result = flow.generate_answer(_CONCEPTUAL)

    assert result.outcome == "not_found"
    assert result.answer == GenerateFlow.NO_EVIDENCE_ANSWER


# --- Facts check (#6): a hedged reply is a refusal only when it gives nothing that was asked. ---

# Hedges, but cites a nearby block that checks out, with good retrieval: before the facts check
# this was "answered" and showed a green badge.
_HEDGED_NEARBY = (
    "The provided context does not specify which object sets were used; it describes "
    "JamPacker's planning and recovery modules. [[src:pdf/jampacker]]"
)
_NO_FACTS = '{"facts": [], "gap_stated": true, "premise_corrected": false}'
_SOME_FACTS = '{"facts": ["JamPacker uses planning and recovery modules"], "gap_stated": true, "premise_corrected": false}'


class FactsCheckLLM(StubLLMClient):
    """Plain chat replies come from `responses`; JSON-schema calls are recorded separately."""

    def __init__(self, responses: list[str], facts_reply: str | Exception) -> None:
        super().__init__(responses=responses)
        self.facts_reply = facts_reply
        self.json_calls: list[tuple[list, dict]] = []

    def chat_json(self, messages, schema):
        self.json_calls.append((messages, schema))
        if isinstance(self.facts_reply, Exception):
            raise self.facts_reply
        return ChatResponse(text=self.facts_reply, raw={})


def _facts_flow(
    reply: str, facts_reply: str | Exception, **settings
) -> tuple[GenerateFlow, FactsCheckLLM]:
    from rca.config.settings import Settings

    llm = FactsCheckLLM([_REWRITE, reply], facts_reply)
    flow = GenerateFlow(
        settings=Settings(**settings), retrieve_flow=StubRetrieveFlow(), llm_client=llm
    )
    return flow, llm


def test_hedged_reply_with_nothing_asked_for_is_a_refusal_even_when_its_citation_checks_out() -> (
    None
):
    from rca.flows.facts_check import FACTS_SCHEMA

    flow, llm = _facts_flow(_HEDGED_NEARBY, _NO_FACTS)

    result = flow.generate_answer(_CONCEPTUAL)

    assert result.outcome == "not_found"
    assert result.citations == []
    assert "[[" not in result.answer and "does not specify" in result.answer
    assert "facts_check_no_answer" in result.failure_labels
    [(messages, schema)] = llm.json_calls
    assert schema == FACTS_SCHEMA
    assert _CONCEPTUAL in messages[-1].content and "does not specify" in messages[-1].content
    stages = {stage.name: stage for stage in result.trace.stages}
    assert list(stages)[-1] == "facts_check"
    # The UI shows the last stage's hit count as "N hits"; the check must not reset it.
    assert stages["facts_check"].hit_count == stages["llm_generate"].hit_count > 0


def test_hedged_reply_that_gives_part_of_the_answer_stays_an_answer() -> None:
    flow, _ = _facts_flow(_HEDGED_NEARBY, _SOME_FACTS)

    result = flow.generate_answer(_CONCEPTUAL)

    assert result.outcome == "answered"
    assert [citation.source_id for citation in result.citations] == ["src:pdf/jampacker"]


@pytest.mark.parametrize(
    "facts_reply",
    ["not json", '{"facts": "x"}', RuntimeError("Ollama timed out")],
    ids=["text", "bad-shape", "error"],
)
def test_failed_facts_check_falls_back_to_the_hedge_rule_and_says_so(facts_reply) -> None:
    flow, _ = _facts_flow(_HEDGED_NEARBY, facts_reply)

    result = flow.generate_answer(_CONCEPTUAL)

    # The old rule keeps a hedged reply whose citation checks out with good retrieval.
    assert result.outcome == "answered"
    assert "facts_check_failed" in result.failure_labels


def test_unhedged_reply_gets_no_facts_check() -> None:
    flow, llm = _facts_flow(
        "JamPacker combines planning and recovery modules. [[src:pdf/jampacker]]", _NO_FACTS
    )

    result = flow.generate_answer(_CONCEPTUAL)

    assert result.outcome == "answered"
    assert llm.json_calls == []


def test_facts_check_can_be_turned_off() -> None:
    flow, llm = _facts_flow(_HEDGED_NEARBY, _NO_FACTS, facts_check_enabled=False)

    result = flow.generate_answer(_CONCEPTUAL)

    assert result.outcome == "answered"
    assert llm.json_calls == []
    assert "facts_check_failed" not in result.failure_labels


def test_eval_measures_the_shipped_hedge_rule_and_prompt() -> None:
    from eval import compare_answer_classifiers
    from rca.flows import facts_check

    assert compare_answer_classifiers.is_hedged is facts_check.is_hedged
    assert compare_answer_classifiers.FACTS_SYSTEM is facts_check.FACTS_SYSTEM
    assert compare_answer_classifiers.parse_facts is facts_check.parse_facts


def test_runtime_hedge_rule_matches_everything_the_label_sampler_weighted_up() -> None:
    # The sampler keeps its regex frozen so the label files stay reproducible; the runtime
    # rule must be a superset, or the facts check would skip replies the labels cover.
    import json

    from eval.sample_answer_labels import is_hedged as sampler_hedged
    from rca.flows.facts_check import is_hedged

    answers = [
        item["answer"]
        for name in (
            "answer_labels.json",
            "answer_labels_fulltext_dev.json",
            "answer_labels_fulltext_test.json",
        )
        for item in json.load(open(f"eval/{name}", encoding="utf-8"))["items"]
    ]
    assert [a for a in answers if sampler_hedged(a) and not is_hedged(a)] == []


@pytest.mark.parametrize(
    "reply",
    [
        "The batch size is not provided in the retrieved context.",
        "It does not contain information about the batch size.",
        "The provided excerpts do not contain information about the batch size.",
        "The retrieved passages don't mention the batch size.",
        "I can't find the batch size in your papers.",
    ],
)
@pytest.mark.parametrize("check", [True, False], ids=["check-on", "check-off"])
def test_refusal_phrasings_count_as_hedged_with_the_check_on_or_off(reply, check) -> None:
    # Phrasings the earlier phrase list caught, and close variants, must stay refusals.
    flow, llm = _facts_flow(reply, _NO_FACTS, facts_check_enabled=check)

    result = flow.generate_answer(_CONCEPTUAL)

    assert result.outcome == "not_found"
    assert "llm_abstained" in result.failure_labels
    assert len(llm.json_calls) == (1 if check else 0)


@pytest.mark.parametrize(
    "phrase",
    [
        # The phrase list GenerateFlow used before the facts check; each must stay a hedge.
        "does not contain information",
        "no information",
        "cannot find",
        "not mentioned in",
        "no relevant information",
        "cannot answer",
        "not provided in",
    ],
)
def test_hedge_rule_covers_the_earlier_phrase_list(phrase) -> None:
    from rca.flows.facts_check import is_hedged

    assert is_hedged(f"Sorry, the answer {phrase} the sources.")
