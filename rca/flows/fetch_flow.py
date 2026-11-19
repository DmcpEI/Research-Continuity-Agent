"""Fetch an open-access paper by arXiv ID or DOI (step 18).

Only identifiers leave the machine, with a generic user agent. The PDF is ingested by the
caller exactly like an upload; this module only finds it, downloads it and says where it came
from.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

USER_AGENT = "research-continuity-agent"
MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 300  # a whole download, not one read

_ARXIV_NEW = r"\d{4}\.\d{4,5}(?:v\d+)?"
_ARXIV_OLD = r"[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
_ARXIV = re.compile(
    r"^(?:arxiv:\s*|(?:https?://)?(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/)?"
    rf"({_ARXIV_NEW}|{_ARXIV_OLD})(?:\.pdf)?/?(?:[?#].*)?$",
    re.IGNORECASE,
)
_DOI = re.compile(
    r"^(?:doi:\s*|(?:https?://)?(?:www\.|dx\.)?doi\.org/)?(10\.\d{4,9}/\S+)$", re.IGNORECASE
)
_INVISIBLE = re.compile("[\u200b-\u200d\ufeff]")
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
_ATOM = "{http://www.w3.org/2005/Atom}"
_FORMS = "an arXiv ID (2501.10868, arXiv:2501.10868v2) or a DOI (10.3390/logistics8030069)"


class FetchError(Exception):
    """Why a paper could not be fetched, with the HTTP status the API should answer."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class FetchedPaper(BaseModel):
    pdf: bytes
    file_name: str
    title: str
    origin: dict[str, Any]


def _require_http(url: str) -> None:
    """Links come from outside (OpenAlex records, redirects): never a file:, data: or ftp: URL."""
    if urllib.parse.urlsplit(url).scheme.lower() not in ("http", "https"):
        raise FetchError(502, f"Refused a link that is not http(s): {url[:80]}")


class _HttpOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_http(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpOnlyRedirects)


def http_get(url: str) -> bytes:
    """GET over http(s) with a generic user agent; at most 100 MB and 5 minutes."""
    _require_http(url)
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    blocks, size = [], 0
    with _OPENER.open(request, timeout=60) as response:
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > MAX_PDF_BYTES:
                raise FetchError(502, f"The file at {url} is larger than 100 MB.")
            if time.monotonic() - started > MAX_SECONDS:
                raise FetchError(502, f"The download from {url} took longer than 5 minutes.")
            blocks.append(block)
    return b"".join(blocks)


def parse_identifier(text: str) -> tuple[str, str]:
    """("arxiv", id) or ("doi", doi); FetchError(400) for anything else.

    Cleans what people paste: invisible characters, a "DOI:" label, surrounding brackets and
    punctuation copied from the end of a citation.
    """
    value = _INVISIBLE.sub("", text or "").strip()
    if value.startswith("("):
        value = value[1:]
    while value and (
        value[-1] in ".,;:" or (value[-1] == ")" and value.count(")") > value.count("("))
    ):
        value = value[:-1]
    if match := _ARXIV.match(value):
        return "arxiv", match.group(1)
    if match := _DOI.match(value):
        doi = urllib.parse.unquote(match.group(1))
        if arxiv := _ARXIV_DOI.match(doi):
            return parse_identifier(arxiv.group(1))
        if any(part in (".", "..") for part in doi.split("/")):
            raise FetchError(400, f"Not a DOI: {value!r}.")
        return "doi", doi
    raise FetchError(400, f"Not an arXiv ID or DOI: {value!r}. Use {_FORMS}.")


def fetch_paper(
    text: str,
    get: Callable[[str], bytes] = http_get,
    now: datetime | None = None,
) -> FetchedPaper:
    kind, identifier = parse_identifier(text)
    fetched_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    if kind == "arxiv":
        title, pinned, copies = _arxiv(identifier, get)
    else:
        title, pinned, copies = _doi(identifier, get)
    pdf, pdf_url, license_url = _download_first(copies, get)
    # At most 200 bytes, so the name (plus "-N.pdf") fits the 255-byte limit of most filesystems.
    stem = re.sub(r"\W+", "_", title).strip("_").encode("utf-8")[:200].decode("utf-8", "ignore")
    stem = stem.strip("_") or "paper"
    return FetchedPaper(
        pdf=pdf,
        file_name=f"{stem}.pdf",
        title=title,
        origin={
            "kind": kind,
            "identifier": pinned,
            "url": pdf_url,
            "license": license_url,
            "fetched_at": fetched_at,
        },
    )


def _call(get: Callable[[str], bytes], url: str) -> bytes:
    try:
        return get(url)
    except FetchError:
        raise
    except Exception as exc:  # network, HTTP status, timeout
        raise FetchError(
            502, f"Could not reach {urllib.parse.urlsplit(url).netloc}: {exc}"
        ) from exc


def _arxiv(identifier: str, get) -> tuple[str, str, list[tuple[str, str | None]]]:
    feed = _call(get, f"https://export.arxiv.org/api/query?id_list={identifier}")
    try:
        entry = ElementTree.fromstring(feed).find(f"{_ATOM}entry")
    except ElementTree.ParseError as exc:
        raise FetchError(502, f"arXiv sent a reply that is not a feed: {exc}") from exc
    entry_id = entry.findtext(f"{_ATOM}id", "") if entry is not None else ""
    if entry is None or "/abs/" not in entry_id:
        raise FetchError(404, f"arXiv has no paper {identifier}.")
    pinned = entry_id.rsplit("/abs/", 1)[1]
    title = " ".join(entry.findtext(f"{_ATOM}title", "").split()) or pinned
    base = re.sub(r"v\d+$", "", pinned)
    license_url = None
    try:  # the licence is only in OAI-PMH; record it as unknown if that call fails
        record = get(
            "https://export.arxiv.org/oai2?verb=GetRecord"
            f"&identifier=oai:arXiv.org:{base}&metadataPrefix=arXiv"
        ).decode("utf-8", "replace")
        if match := re.search(r"<license>\s*([^<\s]+)\s*</license>", record):
            license_url = match.group(1)
    except Exception:
        pass
    return title, pinned, [(f"https://arxiv.org/pdf/{pinned}", license_url)]


def _doi(doi: str, get) -> tuple[str, str, list[tuple[str, str | None]]]:
    url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='/:;()._-')}"
    try:
        raw = get(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FetchError(404, f"OpenAlex has no record of DOI {doi}.") from exc
        raise FetchError(502, f"OpenAlex answered HTTP {exc.code} for DOI {doi}.") from exc
    except FetchError:
        raise
    except Exception as exc:  # network, timeout
        raise FetchError(502, f"Could not reach api.openalex.org: {exc}") from exc
    try:
        work = json.loads(raw)
    except ValueError as exc:
        raise FetchError(502, f"OpenAlex sent a reply that is not JSON: {exc}") from exc
    if not isinstance(work, dict):
        raise FetchError(502, "OpenAlex sent a reply that is not a work record.")
    # The best open copy first, then every other open copy OpenAlex knows, once each: some
    # publishers refuse downloads that do not come from a browser.
    copies: dict[str, str | None] = {}
    locations = work.get("locations")
    for location in [work.get("best_oa_location")] + (
        locations if isinstance(locations, list) else []
    ):
        if isinstance(location, dict) and location.get("pdf_url"):
            copies.setdefault(location["pdf_url"], location.get("license"))
    if not copies:
        raise FetchError(404, f"No open-access PDF found for DOI {doi}.")
    title = " ".join(str(work.get("title") or doi).split())
    return title, doi, list(copies.items())


def _download_first(copies: list[tuple[str, str | None]], get) -> tuple[bytes, str, str | None]:
    """The first copy that downloads as a PDF: (bytes, its URL, its licence)."""
    problems = []
    for url, license_url in copies:
        try:
            return _download_pdf(url, get), url, license_url
        except FetchError as exc:
            problems.append(str(exc))
    raise FetchError(
        502,
        " ".join(problems)
        + f" The PDF is at {copies[0][0]}: open it in your browser and upload it.",
    )


def _download_pdf(url: str, get) -> bytes:
    _require_http(url)
    body = _call(get, url)
    if not body.startswith(b"%PDF"):
        start = body[:80].decode("utf-8", "replace").strip()
        raise FetchError(502, f"{url} sent something that is not a PDF: {start!r}")
    return body
