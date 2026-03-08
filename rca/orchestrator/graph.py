"""Thin orchestration layer over flows and router logic."""

from __future__ import annotations

from rca.flows.generate_flow import GenerateFlow
from rca.flows.ingest_flow import IngestFlow
from rca.flows.retrieve_flow import RetrieveFlow
from rca.orchestrator.routing import Router
from rca.orchestrator.state import OrchestratorState


class ResearchContinuityGraph:
    """Sequential orchestrator that can be replaced with LangGraph later."""

    def __init__(
        self,
        router: Router | None = None,
        ingest_flow: IngestFlow | None = None,
        retrieve_flow: RetrieveFlow | None = None,
        generate_flow: GenerateFlow | None = None,
    ) -> None:
        self.router = router or Router()
        self.ingest_flow = ingest_flow or IngestFlow()
        self.retrieve_flow = retrieve_flow or RetrieveFlow()
        self.generate_flow = generate_flow or GenerateFlow()

    def invoke(self, state: OrchestratorState) -> OrchestratorState:
        if not state.user_query:
            state.errors.append("user_query is required")
            return state

        intent = self.router.detect_intent(state.user_query)
        flow_name = self.router.flow_for_intent(intent)
        state.route = flow_name

        if state.requested_tools:
            state.allowed_tools = self.router.enforce_tool_policy(flow_name, state.requested_tools)

        if flow_name == "ingest_flow":
            target = state.final_output.get("path") or state.user_query.split(maxsplit=1)[-1]
            result = self.ingest_flow.ingest_path(target)
            state.final_output = result.model_dump()
            return state

        if flow_name == "retrieve_flow":
            bundle = self.retrieve_flow.retrieve(state.user_query)
            state.retrieved_hits = [hit.model_dump() for hit in bundle.hits]
            state.final_output = bundle.model_dump(mode="json")
            return state

        try:
            answer = self.generate_flow.generate_answer(state.user_query)
            state.final_output = {"answer": answer}
        except NotImplementedError as exc:
            state.errors.append(str(exc))
        return state


def build_langgraph() -> object | None:
    """Return a LangGraph builder when the optional dependency is installed."""

    try:
        from langgraph.graph import StateGraph
    except ImportError:
        return None
    return StateGraph
