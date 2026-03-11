"""Filesystem MCP server — exposes file tools to agents."""

from __future__ import annotations

import subprocess
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from pydantic import BaseModel


# --- Data models ---

class FileInfo(BaseModel):
    path: str
    is_dir: bool
    size: int


# --- Core logic class ---

MAX_FILE_BYTES = 1_000_000

class FilesystemServer:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def list_directory(self, path: str = ".") -> list[FileInfo]:
        target = self._resolve(path)
        return [
            FileInfo(
                path=str(item.relative_to(self.root)),
                is_dir=item.is_dir(),
                size=item.stat().st_size,
            )
            for item in sorted(target.iterdir())
        ]

    def read_text_file(self, path: str) -> str:
        target = self._resolve(path)
        if target.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"File too large: {target.stat().st_size} bytes (limit {MAX_FILE_BYTES})")
        return target.read_text(encoding="utf-8")

    def search_text(self, pattern: str, path: str = ".") -> list[str]:
        """ripgrep search — returns matching lines with file:line:content format."""
        target = self._resolve(path)
        result = subprocess.run(
            ["rg", "--line-number", "--with-filename", pattern, str(target)],
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()

    def _resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise PermissionError(f"Path escapes root: {path}")
        return candidate


# --- MCP server wiring ---

def make_server(root: str) -> Server:
    fs = FilesystemServer(root)
    server = Server("filesystem")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """Tell MCP clients what tools this server exposes."""
        return [
            types.Tool(
                name="list_directory",
                description="List files and folders inside the research root",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                    },
                },
            ),
            types.Tool(
                name="read_text_file",
                description="Read a text file (max 1MB). Returns full content.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="search_text",
                description="Search for a pattern in files using ripgrep",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                    },
                    "required": ["pattern"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent]:
        """Route tool calls to the right method."""
        try:
            if name == "list_directory":
                result = fs.list_directory(arguments.get("path", "."))
                text = "\n".join(
                    f"{'[DIR]' if f.is_dir else '[FILE]'} {f.path} ({f.size}b)"
                    for f in result
                )
            elif name == "read_text_file":
                text = fs.read_text_file(arguments["path"])
            elif name == "search_text":
                lines = fs.search_text(
                    arguments["pattern"],
                    arguments.get("path", "."),
                )
                text = "\n".join(lines) if lines else "No matches found."
            else:
                text = f"Unknown tool: {name}"
        except (PermissionError, ValueError) as e:
            text = f"Error: {e}"

        return [types.TextContent(type="text", text=text)]

    return server


# --- Entry point ---

async def main(root: str) -> None:
    server = make_server(root)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    import sys

    root_path = sys.argv[1] if len(sys.argv) > 1 else "."
    asyncio.run(main(root_path))