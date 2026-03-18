"""Tool registry for the RCA agent loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from time import perf_counter
from typing import Any

import mcp.types as mcp_types

from rca.agent.contracts import ToolCallStatus, ToolCallTrace
from rca.agent.mcp import MCPClientManager
from rca.config.settings import Settings, get_settings
from rca.flows.retrieve_flow import RetrieveFlow


class KnowledgeBaseAdapter:
    """Native adapter around RetrieveFlow for agent tool use."""

    def __init__(
        self,
        settings: Settings | None = None,
        retrieve_flow: RetrieveFlow | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieve_flow = retrieve_flow or RetrieveFlow(settings=self.settings)

    def search(self, query: str, limit: int = 5) -> str:
        bundle = self.retrieve_flow.retrieve(query, limit=limit)
        if not bundle.hits:
            return "No relevant documents found."

        lines: list[str] = []
        for hit in bundle.hits:
            lines.append(f"[{hit.node_id}] {hit.title} (score={hit.score:.3f})")
            lines.append(hit.excerpt[:300])
            lines.append("")
        return "\n".join(lines).strip()


class ToolRegistry:
    """Unified tool registry: native knowledge base + MCP tools."""

    def __init__(
        self,
        settings: Settings | None = None,
        knowledge_base_search: Callable[[str, int], str] | None = None,
        mcp_manager: MCPClientManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if knowledge_base_search is None:
            self._knowledge_base_search = KnowledgeBaseAdapter(self.settings).search
        else:
            self._knowledge_base_search = knowledge_base_search
        self._mcp_manager = mcp_manager or MCPClientManager(self.settings)
        self._tools: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable[..., str]] = {}
        self._mcp_loaded = False
        self._register_knowledge_base()

    def ollama_tool_definitions(self) -> list[dict[str, Any]]:
        self._ensure_mcp_tools_loaded()
        return list(self._tools.values())

    def call(self, tool_name: str, arguments: dict[str, Any]) -> ToolCallTrace:
        self._ensure_mcp_tools_loaded()
        started = perf_counter()
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolCallTrace(
                tool_name=tool_name,
                input=arguments,
                output=f"Unknown tool: {tool_name}",
                status=ToolCallStatus.error,
                duration_ms=(perf_counter() - started) * 1000.0,
            )

        try:
            result = handler(**arguments)
            output = result if isinstance(result, str) else json.dumps(result, default=str)
            status = ToolCallStatus.success
        except Exception as exc:
            output = f"Error: {exc}"
            status = ToolCallStatus.error

        return ToolCallTrace(
            tool_name=tool_name,
            input=arguments,
            output=output,
            status=status,
            duration_ms=(perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        self._mcp_manager.close()

    def _register_knowledge_base(self) -> None:
        self._tools["search_knowledge_base"] = {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": (
                    "Search the RCA research knowledge base for relevant papers and excerpts. "
                    "Use this before answering research questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
        self._handlers["search_knowledge_base"] = self._knowledge_base_search

    def _ensure_mcp_tools_loaded(self) -> None:
        if self._mcp_loaded:
            return

        for server_name in self._enabled_server_names():
            for tool in self._mcp_manager.list_tools(server_name):
                self._tools[tool.name] = self._to_ollama_tool_definition(tool)
                self._handlers[tool.name] = self._build_mcp_handler(tool.name)

        self._mcp_loaded = True

    def _enabled_server_names(self) -> tuple[str, ...]:
        servers = ["experiments"]
        if self.settings.enable_filesystem_tools:
            servers.insert(0, "filesystem")
        return tuple(servers)

    def _build_mcp_handler(self, tool_name: str) -> Callable[..., str]:
        def handler(**arguments: Any) -> str:
            return self._mcp_manager.call_tool(tool_name, arguments)

        return handler

    @staticmethod
    def _to_ollama_tool_definition(tool: mcp_types.Tool) -> dict[str, Any]:
        parameters = tool.inputSchema or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": parameters,
            },
        }
