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


def test_search_text_passes_pattern_as_data_not_as_rg_flags(tmp_path, monkeypatch) -> None:
    import subprocess

    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    server = FilesystemServer(tmp_path)
    server.search_text("--pre=/bin/sh")

    argv = captured["argv"]
    # pattern only ever follows -e, and the path follows --, so neither can become a flag
    assert argv[argv.index("-e") + 1] == "--pre=/bin/sh"
    assert argv.index("--") == len(argv) - 2
    assert "--pre=/bin/sh" not in argv[: argv.index("-e")]


def test_search_text_does_not_execute_pre_command_from_pattern(tmp_path) -> None:
    import shutil

    import pytest

    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "a.txt").write_text("literal --pre=marker line\n", encoding="utf-8")
    marker = tmp_path / "PWNED"
    script = tmp_path / "pre.sh"
    script.write_text(f'#!/bin/sh\ntouch "{marker}"\ncat "$1"\n', encoding="utf-8")
    script.chmod(0o755)

    server = FilesystemServer(tmp_path / "root")
    matches = server.search_text(f"--pre={script}")

    assert not marker.exists()
    assert matches == []  # searched literally; no line contains the script path
    assert server.search_text("--pre=marker") == [
        f"{tmp_path / 'root' / 'a.txt'}:1:literal --pre=marker line"
    ]
