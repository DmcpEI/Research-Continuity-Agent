"""Tiny PDFs built by hand for tests: pages with a text layer, or blank like a scan."""

from __future__ import annotations


def make_pdf(pages: list[str | None]) -> bytes:
    """One page per item: a string becomes that page's text, None a page with no text layer."""
    objects: list[bytes] = []
    kids = []
    page_numbers = []
    font_number = 3 + 2 * len(pages)
    for index, text in enumerate(pages):
        page_number, content_number = 3 + 2 * index, 4 + 2 * index
        page_numbers.append(page_number)
        kids.append(f"{page_number} 0 R".encode())
        stream = f"BT /F1 10 Tf 40 700 Td ({text}) Tj ET".encode() if text else b""
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R"
            b" /Resources << /Font << /F1 %d 0 R >> >> >>" % (content_number, font_number)
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    head = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [" + b" ".join(kids) + b"] /Count %d >>" % len(pages),
    ]
    body = head + objects + [b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(body, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(body) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(body) + 1, xref)
    return out
