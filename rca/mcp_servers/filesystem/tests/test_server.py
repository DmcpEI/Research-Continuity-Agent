from __future__ import annotations

import pytest

from rca.mcp_servers.filesystem.server import FilesystemServer


def test_filesystem_server_lists_and_reads_files(tmp_path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    server = FilesystemServer(tmp_path)

    listing = server.list_directory()
    content = server.read_text_file("note.txt")

    assert listing[0].path == "note.txt"
    assert content == "hello"


def test_filesystem_server_blocks_path_escape(tmp_path) -> None:
    server = FilesystemServer(tmp_path)

    with pytest.raises(PermissionError):
        server.read_text_file("../secret.txt")
