from cli import rca_ingest


def test_cli_says_why_a_file_with_no_text_is_refused(tmp_path, capsys) -> None:
    from tests.pdf_fixtures import make_pdf

    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf([None]))

    assert rca_ingest.main([str(path)]) == 1

    err = capsys.readouterr().err
    assert "No text found in scan.pdf" in err and "Traceback" not in err
