"""Demo filesystem server facade."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class FileInfo(BaseModel):
    path: str
    is_dir: bool
    size: int


class FilesystemServer:
    """Constrain file operations to a configured root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def list_directory(self, path: str = ".") -> list[FileInfo]:
        target = self._resolve(path)
        return [
            FileInfo(path=str(item.relative_to(self.root)), is_dir=item.is_dir(), size=item.stat().st_size)
            for item in sorted(target.iterdir())
        ]

    def read_text_file(self, path: str) -> str:
        target = self._resolve(path)
        return target.read_text(encoding="utf-8")

    def stat_path(self, path: str) -> FileInfo:
        target = self._resolve(path)
        return FileInfo(path=str(target.relative_to(self.root)), is_dir=target.is_dir(), size=target.stat().st_size)

    def _resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise PermissionError(f"Path escapes configured root: {path}")
        return candidate
