from datetime import UTC, datetime

import pytest

from rca.flows.fetch_flow import FetchError, fetch_paper, parse_identifier

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
PDF = b"%PDF-1.7 paper bytes"

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.10868v3</id>
    <title>JSONSchemaBench: A Rigorous
      Benchmark of Structured Outputs for Language Models</title>
  </entry>
</feed>"""

ATOM_ERROR = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_9999.99999</id>
    <title>Error</title>
  </entry>
</feed>"""

ATOM_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

OAI = "<OAI-PMH><license>http://creativecommons.org/licenses/by/4.0/</license></OAI-PMH>"

API = "https://export.arxiv.org/api/query?id_list=2501.10868"
OAI_URL = (
    "https://export.arxiv.org/oai2?verb=GetRecord"
    "&identifier=oai:arXiv.org:2501.10868&metadataPrefix=arXiv"
)
ARXIV_PDF = "https://arxiv.org/pdf/2501.10868v3"


class FakeWeb:
    """URL -> bytes, or an exception to raise; records every URL asked for."""

    def __init__(self, pages: dict[str, bytes | str | Exception]) -> None:
        self.pages = pages
        self.asked: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.asked.append(url)
        page = self.pages.get(url, OSError(f"404 for {url}"))
        if isinstance(page, Exception):
            raise page
        return page.encode() if isinstance(page, str) else page


def _arxiv_web(**overrides) -> FakeWeb:
    return FakeWeb({API: ATOM, OAI_URL: OAI, ARXIV_PDF: PDF, **overrides})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2501.10868", ("arxiv", "2501.10868")),
        (" arXiv:2501.10868v2 ", ("arxiv", "2501.10868v2")),
        ("https://arxiv.org/abs/2501.10868v3", ("arxiv", "2501.10868v3")),
        ("https://arxiv.org/pdf/2501.10868", ("arxiv", "2501.10868")),
        ("cs/0112017", ("arxiv", "cs/0112017")),
        ("10.3390/logistics8030069", ("doi", "10.3390/logistics8030069")),
        ("doi:10.1109/LRA.2020.3043168", ("doi", "10.1109/LRA.2020.3043168")),
        ("https://doi.org/10.1109/LRA.2020.3043168", ("doi", "10.1109/LRA.2020.3043168")),
        ("10.48550/arXiv.2501.10868", ("arxiv", "2501.10868")),  # arXiv's own DOI
    ],
)
def test_parse_identifier_accepts_the_usual_forms(text, expected):
    assert parse_identifier(text) == expected


@pytest.mark.parametrize(
    "text", ["", "   ", "JSONSchemaBench", "2501", "10.1234", "http://x.org/a"]
)
def test_parse_identifier_rejects_anything_else(text):
    with pytest.raises(FetchError) as error:
        parse_identifier(text)
    assert error.value.status == 400
    assert "2501.10868" in str(error.value)  # the message shows what is accepted


def test_arxiv_fetch_pins_the_version_and_records_the_licence():
    web = _arxiv_web()

    paper = fetch_paper("2501.10868", get=web, now=NOW)

    assert paper.pdf == PDF
    assert paper.title == (
        "JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models"
    )
    assert paper.file_name == (
        "JSONSchemaBench_A_Rigorous_Benchmark_of_Structured_Outputs_for_Language_Models.pdf"
    )
    assert paper.origin == {
        "kind": "arxiv",
        "identifier": "2501.10868v3",
        "url": ARXIV_PDF,
        "license": "http://creativecommons.org/licenses/by/4.0/",
        "fetched_at": "2026-11-19T12:00:00+00:00",
    }


def test_arxiv_licence_is_unknown_when_its_lookup_fails():
    paper = fetch_paper("2501.10868", get=_arxiv_web(**{OAI_URL: OSError("503")}), now=NOW)

    assert paper.pdf == PDF
    assert paper.origin["license"] is None


@pytest.mark.parametrize("feed", [ATOM_ERROR, ATOM_EMPTY])
def test_an_unknown_arxiv_id_is_a_404_and_downloads_nothing(feed):
    web = _arxiv_web(**{API: feed})

    with pytest.raises(FetchError) as error:
        fetch_paper("2501.10868", get=web, now=NOW)

    assert error.value.status == 404
    assert ARXIV_PDF not in web.asked


def test_open_doi_fetches_the_open_access_pdf():
    doi_api = "https://api.openalex.org/works/doi:10.3390/logistics8030069"
    oa_pdf = "https://www.mdpi.com/2305-6290/8/3/69/pdf"
    work = (
        '{"title": "Comprehensive Review of Robotized Freight Packing",'
        f' "best_oa_location": {{"pdf_url": "{oa_pdf}", "license": "cc-by"}}}}'
    )
    paper = fetch_paper(
        "10.3390/logistics8030069", get=FakeWeb({doi_api: work, oa_pdf: PDF}), now=NOW
    )

    assert paper.pdf == PDF
    assert paper.file_name == "Comprehensive_Review_of_Robotized_Freight_Packing.pdf"
    assert paper.origin == {
        "kind": "doi",
        "identifier": "10.3390/logistics8030069",
        "url": oa_pdf,
        "license": "cc-by",
        "fetched_at": "2026-11-19T12:00:00+00:00",
    }


def test_closed_doi_is_a_404_and_downloads_nothing():
    doi_api = "https://api.openalex.org/works/doi:10.1109/LRA.2020.3043168"
    web = FakeWeb({doi_api: '{"title": "Jampacker", "best_oa_location": null}'})

    with pytest.raises(FetchError) as error:
        fetch_paper("10.1109/LRA.2020.3043168", get=web, now=NOW)

    assert error.value.status == 404
    assert "open-access" in str(error.value)
    assert web.asked == [doi_api]


def test_unknown_doi_is_a_404():
    with pytest.raises(FetchError) as error:
        fetch_paper("10.9999/nothing", get=FakeWeb({}), now=NOW)
    assert error.value.status in (404, 502)


def test_a_page_that_is_not_a_pdf_is_refused():
    web = _arxiv_web(**{ARXIV_PDF: b"<html>Please verify you are human</html>"})

    with pytest.raises(FetchError) as error:
        fetch_paper("2501.10868", get=web, now=NOW)

    assert error.value.status == 502
    assert "not a PDF" in str(error.value)


def test_network_failure_is_a_502():
    with pytest.raises(FetchError) as error:
        fetch_paper("2501.10868", get=_arxiv_web(**{API: OSError("timed out")}), now=NOW)
    assert error.value.status == 502


DOI_API = "https://api.openalex.org/works/doi:10.3390/logistics8030069"
BLOCKED = "https://www.mdpi.com/2305-6290/8/3/69/pdf"
MIRROR = "https://repository.example.org/review.pdf"
WORK_TWO_COPIES = (
    '{"title": "Comprehensive Review of Robotized Freight Packing",'
    f' "best_oa_location": {{"pdf_url": "{BLOCKED}", "license": "cc-by"}},'
    f' "locations": [{{"pdf_url": "{BLOCKED}", "license": "cc-by", "is_oa": true}},'
    ' {"pdf_url": null, "is_oa": false},'
    f' {{"pdf_url": "{MIRROR}", "license": "cc-by-sa", "is_oa": true}}]}}'
)


def test_doi_falls_back_to_another_open_copy_when_the_first_host_refuses():
    web = FakeWeb({DOI_API: WORK_TWO_COPIES, BLOCKED: OSError("HTTP Error 403"), MIRROR: PDF})

    paper = fetch_paper("10.3390/logistics8030069", get=web, now=NOW)

    assert paper.pdf == PDF
    assert paper.origin["url"] == MIRROR
    assert paper.origin["license"] == "cc-by-sa"
    assert web.asked == [DOI_API, BLOCKED, MIRROR]


def test_doi_says_where_the_pdf_is_when_every_host_refuses():
    web = FakeWeb({DOI_API: WORK_TWO_COPIES, BLOCKED: OSError("HTTP Error 403: Forbidden")})

    with pytest.raises(FetchError) as error:
        fetch_paper("10.3390/logistics8030069", get=web, now=NOW)

    assert error.value.status == 502
    message = str(error.value)
    assert BLOCKED in message and "upload" in message


@pytest.mark.parametrize(
    "bad_url", ["file:///etc/passwd", "data:application/pdf,%25PDF-1.4", "ftp://host/a.pdf"]
)
def test_only_http_links_are_ever_opened(bad_url):
    """An OpenAlex record is outside data: a file:// link must not read a local file."""
    work = (
        '{"title": "T", "locations": ['
        f'{{"pdf_url": "{bad_url}"}}, {{"pdf_url": "https://ok.example.org/t.pdf"}}]}}'
    )
    web = FakeWeb({DOI_API: work, "https://ok.example.org/t.pdf": PDF})

    paper = fetch_paper("10.3390/logistics8030069", get=web, now=NOW)

    assert paper.origin["url"] == "https://ok.example.org/t.pdf"
    assert bad_url not in web.asked


def test_http_get_refuses_other_schemes(tmp_path):
    from rca.flows.fetch_flow import http_get

    secret = tmp_path / "secret.pdf"
    secret.write_bytes(b"%PDF-1.4 local secret")

    with pytest.raises(FetchError):
        http_get(secret.as_uri())


@pytest.mark.parametrize("locations", ["5", '"abc"', '{"pdf_url": "x"}'])
def test_a_malformed_locations_field_is_ignored(locations):
    web = FakeWeb({DOI_API: f'{{"title": "T", "locations": {locations}}}'})

    with pytest.raises(FetchError) as error:
        fetch_paper("10.3390/logistics8030069", get=web, now=NOW)

    assert error.value.status == 404


def test_a_doi_lookup_that_cannot_reach_openalex_is_a_502_not_a_404():
    web = FakeWeb({DOI_API: OSError("timed out")})

    with pytest.raises(FetchError) as error:
        fetch_paper("10.3390/logistics8030069", get=web, now=NOW)

    assert error.value.status == 502


def test_a_doi_openalex_does_not_know_is_a_404():
    import urllib.error

    missing = urllib.error.HTTPError(DOI_API, 404, "Not Found", None, None)
    with pytest.raises(FetchError) as error:
        fetch_paper("10.3390/logistics8030069", get=FakeWeb({DOI_API: missing}), now=NOW)
    assert error.value.status == 404


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10.1000/xyz.", ("doi", "10.1000/xyz")),  # punctuation copied from a citation
        ("10.1000/xyz,", ("doi", "10.1000/xyz")),
        ("(10.1000/xyz)", ("doi", "10.1000/xyz")),
        ("10.1000/xyz​", ("doi", "10.1000/xyz")),
        ("DOI: 10.1000/xyz", ("doi", "10.1000/xyz")),
        ("https://www.doi.org/10.1000/a%2Fb", ("doi", "10.1000/a/b")),
        ("arxiv.org/abs/2501.10868", ("arxiv", "2501.10868")),
        ("https://arxiv.org/abs/2501.10868v2?context=cs", ("arxiv", "2501.10868v2")),
        ("10.1016/s0140-6736(20)30183-5", ("doi", "10.1016/s0140-6736(20)30183-5")),
    ],
)
def test_parse_identifier_cleans_what_people_paste(text, expected):
    assert parse_identifier(text) == expected


@pytest.mark.parametrize("text", ["10.1000/../../authors/A1", "10.1000/./x"])
def test_parse_identifier_rejects_path_tricks(text):
    with pytest.raises(FetchError) as error:
        parse_identifier(text)
    assert error.value.status == 400


def test_file_name_fits_255_bytes_for_long_unicode_titles():
    title = "論文" * 200
    atom = ATOM.replace(
        "JSONSchemaBench: A Rigorous\n      Benchmark of Structured Outputs for Language Models",
        title,
    )

    paper = fetch_paper("2501.10868", get=_arxiv_web(**{API: atom}), now=NOW)

    assert len(paper.file_name.encode("utf-8")) <= 255
    assert paper.file_name.endswith(".pdf")
