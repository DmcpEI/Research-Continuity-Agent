from __future__ import annotations

from rca.flows.generate_flow import GenerateFlow
from rca.flows.retrieve_flow import RetrievalBundle, RetrievalHit
from rca.llm.client import ChatResponse, LLMClient


class StubLLMClient(LLMClient):
    def __init__(self) -> None:
        self.model = "stub-model"
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(
                text="JamPacker components planning recovery packing efficiency",
                raw={"prompt_eval_count": 11, "eval_count": 4},
            )
        return ChatResponse(
            text="JamPacker combines planning and recovery modules. [[src:pdf/jampacker]]",
            raw={"prompt_eval_count": 29, "eval_count": 8},
        )

    def embed(self, texts: list[str], dimensions: int = 32) -> list[list[float]]:
        raise NotImplementedError


class StubRetrieveFlow:
    def retrieve(self, query: str, limit: int = 10, trace=None) -> RetrievalBundle:
        return RetrievalBundle(
            query=query,
            hits=[
                RetrievalHit(
                    node_id="src:pdf/jampacker",
                    score=0.92,
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

    assert result.trace is not None
    assert result.trace.model == "stub-model"
    assert result.trace.rewritten_query is not None
    assert result.trace.context_node_ids == ["src:pdf/jampacker"]
    assert [stage.name for stage in result.trace.stages] == ["llm_rewrite", "llm_generate"]
    assert result.trace.prompt_tokens == 40
    assert result.trace.completion_tokens == 12
    assert result.trace.total_latency_ms > 0.0
