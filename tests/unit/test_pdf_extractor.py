from rca.extractors.pdf_extractor import PDFExtractor
from tests.pdf_fixtures import make_pdf


def test_pdf_extractor_counts_pages_without_text(tmp_path) -> None:
    path = tmp_path / "partly_scanned.pdf"
    path.write_bytes(make_pdf(["A cover page with text.", None, None]))

    payload = PDFExtractor().extract(path)

    assert "cover page" in payload["content"]
    assert payload["metadata"]["page_count"] == 3
    assert payload["metadata"]["pages_without_text"] == 2


def test_pdf_extractor_reports_zero_for_a_normal_pdf(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf(["Page one.", "Page two."]))

    assert PDFExtractor().extract(path)["metadata"]["pages_without_text"] == 0
