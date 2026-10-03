"""Tool registry for the RCA agent loop."""

from __future__ import annotations

import json
import re
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

        best_hit_by_source: dict[str, Any] = {}
        for hit in bundle.hits:
            source_id = self._to_source_id(hit.node_id)
            current = best_hit_by_source.get(source_id)
            if current is None or hit.score > current.score:
                best_hit_by_source[source_id] = hit

        unique_hits = sorted(
            best_hit_by_source.values(),
            key=lambda item: item.score,
            reverse=True,
        )[:limit]

        lines: list[str] = []
        for index, hit in enumerate(unique_hits, start=1):
            source_id = self._to_source_id(hit.node_id)
            lines.append(f"RESULT {index}")
            lines.append(f"source_id: {source_id}")
            lines.append(f"title: {hit.title}")
            lines.append(f"score: {hit.score:.3f}")
            lines.append(f"excerpt: {hit.excerpt[:300]}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _to_source_id(node_id: str) -> str:
        if node_id.startswith("src:"):
            return node_id
        if node_id.startswith("chk:"):
            return re.sub(r":\d+$", "", node_id.replace("chk:", "src:", 1))
        return node_id


# Search results expose these IDs; models sometimes treat them as file paths.
KNOWLEDGE_BASE_ID_PREFIXES = ("src:", "chk:")
FILESYSTEM_PATH_TOOLS = frozenset({"read_text_file", "list_directory", "search_text"})
MAX_ECHOED_ID = 200


def _as_knowledge_base_id(path: Any) -> str | None:
    """The source/chunk ID a path argument really is, also in [[citation]] form, else None."""
    if not isinstance(path, str):
        return None
    candidate = path.strip().removeprefix("[[").removesuffix("]]").strip()
    if not candidate.startswith(KNOWLEDGE_BASE_ID_PREFIXES):
        return None
    return candidate[:MAX_ECHOED_ID]


class ToolRegistry:
    """Unified tool registry: native knowledge base + MCP tools."""

    def __init__(
        self,
        settings: Settings | None = None,
        knowledge_base_search: Callable[[str, int], str] | None = None,
        mcp_manager: MCPClientManager | None = None,
        retrieve_flow: RetrieveFlow | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if knowledge_base_search is not None and retrieve_flow is not None:
            raise ValueError("Pass knowledge_base_search or retrieve_flow, not both")
        if knowledge_base_search is None:
            # Pass the caller's retrieve flow so agent search shares its vector store.
            self._knowledge_base_search = KnowledgeBaseAdapter(
                self.settings, retrieve_flow=retrieve_flow
            ).search
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

        # Path guards: redirect calls that cannot work. They report status error so the
        # agent eval counts them as invalid; the model only sees the guidance text.
        raw_path = arguments.get("path")
        if tool_name in FILESYSTEM_PATH_TOOLS and tool_name in self._handlers:
            source_id = _as_knowledge_base_id(raw_path)
            if source_id is not None:
                cite = KnowledgeBaseAdapter._to_source_id(source_id)
                return ToolCallTrace(
                    tool_name=tool_name,
                    input=arguments,
                    output=(
                        f"'{source_id}' is a knowledge-base source ID, not a file path. "
                        "Answer from the `search_knowledge_base` excerpts and cite it as "
                        f"[[{cite}]]; run another `search_knowledge_base` query for more detail."
                    ),
                    status=ToolCallStatus.error,
                    duration_ms=(perf_counter() - started) * 1000.0,
                )
            if (
                tool_name == "read_text_file"
                and isinstance(raw_path, str)
                and raw_path.strip().lower().endswith(".pdf")
            ):
                return ToolCallTrace(
                    tool_name=tool_name,
                    input=arguments,
                    output=(
                        "This is a PDF file. Use `search_knowledge_base` to query its content "
                        "instead of reading it directly."
                    ),
                    status=ToolCallStatus.error,
                    duration_ms=(perf_counter() - started) * 1000.0,
                )

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
                    "Use this before answering research questions. Results carry source_id "
                    "values (src:...) that are citation IDs, not file paths: answer from the "
                    "excerpts and cite them as [[source_id]]."
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
