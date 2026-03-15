from __future__ import annotations

from rca.flows.generate_flow import GenerateFlow
from rca.flows.retrieve_flow import RetrievalBundle, RetrievalHit
from rca.llm.client import ChatResponse, LLMClient


class StubLLMClient(LLMClient):
    def __init__(self, responses: list[str] | None = None) -> None:
        self.model = "stub-model"
        self.calls = 0
        self.responses = responses or [
            "JamPacker components planning recovery packing efficiency",
            "JamPacker combines planning and recovery modules. [[src:pdf/jampacker]]",
        ]

    def chat(self, messages):
        self.calls += 1
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

    def retrieve(self, query: str, limit: int = 10, trace=None) -> RetrievalBundle:
        return RetrievalBundle(
            query=query,
            hits=[
                RetrievalHit(
                    node_id="src:pdf/jampacker",
                    score=self.score,
                    title="JamPacker",
                    excerpt="JamPacker uses planning and recovery modules.",
                    metadata={},
                )
            ],
            related_edges=[],
            trace=trace,
        )


def test_generate_answer_attaches_query_trace() -> None:
    flow = GenerateFlow(retrieve_flow=StubRetrieveFlow(), llm_client=StubLLMClient())

    result = flow.generate_answer("What are the two main components of JamPacker?")

    assert result.abstained is False
    assert result.trace is not None
    assert result.trace.model == "stub-model"
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


def test_generate_answer_appends_citation_for_supported_answer_without_model_citation() -> None:
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
    assert result.grounded is True
    assert [citation.source_id for citation in result.citations] == ["src:pdf/jampacker"]
    assert result.answer.endswith("[[src:pdf/jampacker]]")
