"""Deterministic extractor for markdown and text notes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
TAG_PATTERN = re.compile(r"(?<!\w)#([a-zA-Z][a-zA-Z0-9_-]+)")


class NoteExtractor:
    """Extract title, sections, and tags from local notes."""

    def extract(self, path: str | Path) -> dict[str, Any]:
        note_path = Path(path)
        text = note_path.read_text(encoding="utf-8")
        title = self._extract_title(note_path, text)
        sections = self._extract_sections(text)
        tags = sorted({match.group(1).lower() for match in TAG_PATTERN.finditer(text)})

        return {
            "title": title,
            "content": text.strip(),
            "sections": sections,
            "metadata": {"kind": "note", "path": str(note_path), "tags": tags},
        }

    @staticmethod
    def _extract_title(path: Path, text: str) -> str:
        for line in text.splitlines():
            match = HEADING_PATTERN.match(line.strip())
            if match and len(match.group(1)) == 1:
                return match.group(2).strip()
        return path.stem

    @staticmethod
    def _extract_sections(text: str) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        current_heading = "Introduction"
        current_lines: list[str] = []

        for line in text.splitlines():
            match = HEADING_PATTERN.match(line.strip())
            if match:
                if current_lines:
                    sections.append(
                        {"heading": current_heading, "text": "\n".join(current_lines).strip()}
                    )
                    current_lines = []
                current_heading = match.group(2).strip()
                continue
            current_lines.append(line)

        if current_lines:
            sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})

        return [section for section in sections if section["text"]]
