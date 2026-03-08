"""Deterministic PDF text extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PDFExtractor:
    """Extract page text from PDFs without any model inference."""

    def extract(self, path: str | Path) -> dict[str, Any]:
        pdf_path = Path(path)
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF extraction requires the optional 'pypdf' dependency.") from exc

        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        sections: list[dict[str, Any]] = []

        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(text)
                sections.append({"heading": f"Page {index}", "text": text, "page": index})

        return {
            "title": pdf_path.stem,
            "content": "\n\n".join(pages).strip(),
            "sections": sections,
            "metadata": {"kind": "pdf", "path": str(pdf_path), "page_count": len(reader.pages)},
        }
