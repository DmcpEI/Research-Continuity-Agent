"""Thread-backed MCP stdio clients for agent tools."""

from __future__ import annotations

import asyncio
import atexit
import json
import sys
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from rca.config.settings import Settings, get_settings

_REQUEST_TIMEOUT = 30.0
_TOOL_CALL_TIMEOUT = timedelta(seconds=30)


def _call_result_to_text(result: mcp_types.CallToolResult) -> str:
    parts: list[str] = []
    for item in result.content:
        if isinstance(item, mcp_types.TextContent):
            parts.append(item.text)
        elif hasattr(item, "model_dump_json"):
            parts.append(item.model_dump_json(by_alias=True, exclude_none=True))
        else:
            parts.append(str(item))

    if not parts and result.structuredContent is not None:
        return json.dumps(result.structuredContent, indent=2, sort_keys=True, default=str)

    text = "\n".join(part for part in parts if part)
    if result.isError and text and not text.startswith("Error:"):
        return f"Error: {text}"
    return text or ("Error: MCP tool reported an error." if result.isError else "")


class _EventLoopThread:
    def __init__(self, name: str) -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: Exception | None = None

    def start(self, setup_coro_factory) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._ready = threading.Event()
        self._startup_error = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run,
            args=(setup_coro_factory,),
            name=self._name,
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=_REQUEST_TIMEOUT)

        if self._startup_error is not None:
            raise RuntimeError(str(self._startup_error)) from self._startup_error
        if self._loop is None or not self._thread.is_alive():
            raise RuntimeError(f"{self._name} event loop thread failed to start.")

    def _run(self, setup_coro_factory) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(setup_coro_factory())
        except Exception as exc:  # pragma: no cover - startup failure path
            self._startup_error = exc
            self._ready.set()
            self._loop.close()
            return

        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def run(self, coro):
        if self._loop is None:
            raise RuntimeError(f"{self._name} event loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=_REQUEST_TIMEOUT)

    def stop(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class MCPServerClient:
    """Synchronous facade over a single MCP stdio server."""

    def __init__(
        self,
        name: str,
        server_parameters: StdioServerParameters,
    ) -> None:
        self.name = name
        self.server_parameters = server_parameters
        self._runner = _EventLoopThread(f"mcp-{name}")
        self._lock = threading.Lock()
        self._stdio_cm = None
        self._session: ClientSession | None = None

    async def _setup(self) -> None:
        try:
            self._stdio_cm = stdio_client(self.server_parameters)
            read_stream, write_stream = await self._stdio_cm.__aenter__()
            self._session = ClientSession(read_stream, write_stream)
            await self._session.__aenter__()
            await self._session.initialize()
        except Exception:
            await self._shutdown()
            raise

    async def _shutdown(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None

    def _ensure_started(self) -> None:
        with self._lock:
            if self._runner.is_alive and self._session is not None:
                return
            self._runner.start(self._setup)

    async def _list_tools(self) -> list[mcp_types.Tool]:
        if self._session is None:
            raise RuntimeError(f"{self.name} MCP session is not initialized.")

        cursor: str | None = None
        tools: list[mcp_types.Tool] = []
        while True:
            result = await self._session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if cursor is None:
                return tools

    async def _call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        if self._session is None:
            raise RuntimeError(f"{self.name} MCP session is not initialized.")
        return await self._session.call_tool(
            tool_name,
            arguments=arguments,
            read_timeout_seconds=_TOOL_CALL_TIMEOUT,
        )

    def list_tools(self) -> list[mcp_types.Tool]:
        self._ensure_started()
        try:
            return self._runner.run(self._list_tools())
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Failed to list tools from MCP server '{self.name}': {exc}"
            ) from exc

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        self._ensure_started()
        try:
            result = self._runner.run(self._call_tool(tool_name, arguments))
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Failed to call MCP tool '{tool_name}' on server '{self.name}': {exc}"
            ) from exc
        return _call_result_to_text(result)

    def close(self) -> None:
        with self._lock:
            if self._session is not None and self._runner.is_alive:
                try:
                    self._runner.run(self._shutdown())
                except Exception:
                    pass
            self._session = None
            self._stdio_cm = None
            self._runner.stop()


@dataclass(frozen=True)
class _ServerSpec:
    module: str
    args: tuple[str, ...]


class MCPClientManager:
    """Manage MCP stdio clients for the read-only agent tool set."""

    READ_ONLY_TOOLS = {
        "filesystem": {"list_directory", "read_text_file", "search_text"},
        "experiments": {"list_runs", "get_run"},
    }

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._tool_to_server: dict[str, str] = {}
        self._clients = {
            server_name: MCPServerClient(server_name, params)
            for server_name, params in self._server_parameters().items()
        }
        atexit.register(self.close)

    def _server_parameters(self) -> dict[str, StdioServerParameters]:
        workspace_root = self.settings.workspace_root.resolve()
        filesystem_root = self.settings.filesystem_root.resolve()
        experiment_db = self.settings.experiment_db_path.resolve()

        specs = {
            "filesystem": _ServerSpec(
                module="rca.mcp_servers.filesystem.server",
                args=(str(filesystem_root),),
            ),
            "experiments": _ServerSpec(
                module="rca.mcp_servers.experiments.server",
                args=(str(experiment_db),),
            ),
        }

        return {
            name: StdioServerParameters(
                command=sys.executable,
                args=["-m", spec.module, *spec.args],
                cwd=str(workspace_root),
            )
            for name, spec in specs.items()
        }

    def list_tools(self, server_name: str) -> list[mcp_types.Tool]:
        client = self._clients[server_name]
        allowed = self.READ_ONLY_TOOLS[server_name]
        tools = [tool for tool in client.list_tools() if tool.name in allowed]
        for tool in tools:
            self._tool_to_server[tool.name] = server_name
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        server_name = self._tool_to_server.get(tool_name)
        if server_name is None:
            for candidate in self._clients:
                if tool_name in self.READ_ONLY_TOOLS[candidate]:
                    server_name = candidate
                    self._tool_to_server[tool_name] = candidate
                    break
        if server_name is None:
            raise ValueError(f"Unknown MCP tool: {tool_name}")
        return self._clients[server_name].call_tool(tool_name, arguments)

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
