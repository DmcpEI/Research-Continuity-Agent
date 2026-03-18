from __future__ import annotations

import mcp.types as mcp_types

from rca.agent.contracts import ToolCallStatus, ToolCallTrace
from rca.agent.loop import MAX_ITERATIONS, AgentLoop
from rca.agent.tools import ToolRegistry
from rca.config.settings import Settings
from rca.llm.client import ToolChatResponse


class FakeLLM:
    def __init__(self, responses: list[ToolChatResponse]) -> None:
        self.responses = responses
        self.calls = 0
        self.model = "fake-agent-model"

    def chat_with_tools(self, messages, tools) -> ToolChatResponse:
        del messages, tools
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def ollama_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search the KB",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ]

    def call(self, tool_name: str, arguments: dict) -> ToolCallTrace:
        self.calls.append((tool_name, arguments))
        return ToolCallTrace(
            tool_name=tool_name,
            input=arguments,
            output="Bin packing is NP-hard.",
            status=ToolCallStatus.success,
            duration_ms=5.0,
        )

    def close(self) -> None:
        return None


def test_final_answer_on_first_turn() -> None:
    loop = AgentLoop(
        registry=FakeRegistry(),
        llm_client=FakeLLM(
            [
                ToolChatResponse(
                    text="The answer is 42.", tool_calls=[], raw={"prompt_eval_count": 7}
                )
            ]
        ),
    )

    result = loop.run("What is the answer?")

    assert result.answer == "The answer is 42."
    assert result.trace.iterations == 1
    assert result.trace.stopped_reason == "final_answer"
    assert result.trace.tool_calls == []


def test_tool_call_then_answer() -> None:
    registry = FakeRegistry()
    loop = AgentLoop(
        registry=registry,
        llm_client=FakeLLM(
            [
                ToolChatResponse(
                    text="",
                    tool_calls=[
                        {
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": {"query": "bin packing"},
                            }
                        }
                    ],
                    raw={"prompt_eval_count": 9},
                ),
                ToolChatResponse(
                    text="Based on the search results, bin packing is NP-hard.",
                    tool_calls=[],
                    raw={"eval_count": 11},
                ),
            ]
        ),
    )

    result = loop.run("Tell me about bin packing")

    assert "NP-hard" in result.answer
    assert result.trace.iterations == 2
    assert len(result.trace.tool_calls) == 1
    assert registry.calls == [("search_knowledge_base", {"query": "bin packing"})]
    assert result.trace.stopped_reason == "final_answer"


def test_max_iterations_stops_loop() -> None:
    registry = FakeRegistry()
    loop = AgentLoop(
        registry=registry,
        llm_client=FakeLLM(
            [
                ToolChatResponse(
                    text="",
                    tool_calls=[
                        {
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": {"query": "loop forever"},
                            }
                        }
                    ],
                    raw={},
                )
            ]
            * (MAX_ITERATIONS + 2)
        ),
    )

    result = loop.run("Keep searching forever")

    assert result.trace.stopped_reason == "max_iterations"
    assert result.trace.iterations == MAX_ITERATIONS
    assert len(result.trace.tool_calls) == MAX_ITERATIONS


def test_malformed_tool_call_arguments_fall_back_to_plain_text_answer() -> None:
    loop = AgentLoop(
        registry=FakeRegistry(),
        llm_client=FakeLLM(
            [
                ToolChatResponse(
                    text="I found enough context to answer directly.",
                    tool_calls=[
                        {
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": '{"query": "bin packing"',
                            }
                        }
                    ],
                    raw={},
                )
            ]
        ),
    )

    result = loop.run("Tell me about bin packing")

    assert result.answer == "I found enough context to answer directly."
    assert result.trace.tool_calls == []
    assert "malformed" in result.trace.warnings[0]


def test_tool_registry_exposes_read_only_tools_with_fakes() -> None:
    class FakeMCPManager:
        def list_tools(self, server_name: str) -> list[mcp_types.Tool]:
            if server_name == "filesystem":
                return [
                    mcp_types.Tool(
                        name="list_directory",
                        description="List files",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    mcp_types.Tool(
                        name="read_text_file",
                        description="Read text",
                        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
                    ),
                    mcp_types.Tool(
                        name="search_text",
                        description="Search text",
                        inputSchema={
                            "type": "object",
                            "properties": {"pattern": {"type": "string"}},
                        },
                    ),
                ]
            return [
                mcp_types.Tool(
                    name="list_runs",
                    description="List runs",
                    inputSchema={"type": "object", "properties": {}},
                ),
                mcp_types.Tool(
                    name="get_run",
                    description="Get run",
                    inputSchema={"type": "object", "properties": {"run_id": {"type": "string"}}},
                ),
            ]

        def call_tool(self, tool_name: str, arguments: dict) -> str:
            return f"{tool_name}:{arguments}"

        def close(self) -> None:
            return None

    registry = ToolRegistry(
        knowledge_base_search=lambda query, limit=5: f"KB:{query}:{limit}",
        mcp_manager=FakeMCPManager(),
    )

    tool_names = {tool["function"]["name"] for tool in registry.ollama_tool_definitions()}

    assert tool_names == {
        "search_knowledge_base",
        "list_directory",
        "read_text_file",
        "search_text",
        "list_runs",
        "get_run",
    }

    call_trace = registry.call("search_knowledge_base", {"query": "bin packing", "limit": 3})
    assert call_trace.status == ToolCallStatus.success
    assert call_trace.output == "KB:bin packing:3"


def test_tool_registry_can_disable_filesystem_tools() -> None:
    class FakeMCPManager:
        def list_tools(self, server_name: str) -> list[mcp_types.Tool]:
            if server_name == "filesystem":
                return [
                    mcp_types.Tool(
                        name="list_directory",
                        description="List files",
                        inputSchema={"type": "object", "properties": {}},
                    )
                ]
            return [
                mcp_types.Tool(
                    name="list_runs",
                    description="List runs",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]

        def call_tool(self, tool_name: str, arguments: dict) -> str:
            return f"{tool_name}:{arguments}"

        def close(self) -> None:
            return None

    registry = ToolRegistry(
        settings=Settings(enable_filesystem_tools=False),
        knowledge_base_search=lambda query, limit=5: f"KB:{query}:{limit}",
        mcp_manager=FakeMCPManager(),
    )

    tool_names = {tool["function"]["name"] for tool in registry.ollama_tool_definitions()}

    assert tool_names == {"search_knowledge_base", "list_runs"}
