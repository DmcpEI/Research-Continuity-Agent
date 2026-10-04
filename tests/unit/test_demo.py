from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cli import rca_demo
from rca.api.main import create_site


def _dist(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside dist", encoding="utf-8")
    return dist


def _site(tmp_path) -> TestClient:
    api = FastAPI()
    api.get("/health")(lambda: {"status": "ok"})
    return TestClient(create_site(api=api, frontend_dist=_dist(tmp_path)))


def test_site_serves_api_under_prefix_and_frontend_routes(tmp_path) -> None:
    client = _site(tmp_path)

    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/assets/app.js").text == "console.log(1)"
    for route in ("/", "/library", "/agent/history"):  # client-side routes survive a reload
        assert client.get(route).text == "<html>app</html>"
    assert client.get("/assets/missing.js").status_code == 404  # not index.html as JS
    assert client.head("/library").status_code == 200
    assert client.get("/docs").text == "<html>app</html>"  # the API keeps its docs under /api


def test_site_never_serves_files_outside_the_build(tmp_path) -> None:
    client = _site(tmp_path)

    for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/..%2f..%2fsecret.txt"):
        assert "outside dist" not in client.get(path).text
    assert client.get("/assets/%00.js").status_code == 404  # not a 500


def test_site_requires_a_frontend_build(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="npm run build"):
        create_site(api=FastAPI(), frontend_dist=tmp_path / "missing")


def test_download_missing_skips_existing_files_and_rejects_non_pdf(tmp_path, monkeypatch) -> None:
    papers = [
        {"arxiv": "1111.0001v1", "title": "Have", "file": "Have.pdf"},
        {"arxiv": "1111.0002v1", "title": "Need", "file": "Need.pdf"},
    ]
    (tmp_path / "Have.pdf").write_bytes(b"%PDF-existing")
    requested = []

    def fake_urlopen(request, timeout):
        requested.append(request.full_url)
        return io.BytesIO(b"%PDF-1.7 body")

    monkeypatch.setattr(rca_demo.urllib.request, "urlopen", fake_urlopen)
    paths = rca_demo.download_missing(papers, tmp_path)

    assert requested == ["https://arxiv.org/pdf/1111.0002v1"]
    assert [path.read_bytes()[:5] for path in paths] == [b"%PDF-", b"%PDF-"]

    papers.append({"arxiv": "1111.0003v1", "title": "Blocked", "file": "Blocked.pdf"})
    monkeypatch.setattr(
        rca_demo.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b"<html>")
    )
    with pytest.raises(RuntimeError, match="1111.0003v1"):
        rca_demo.download_missing(papers, tmp_path)
    assert not (tmp_path / "Blocked.pdf").exists()


def test_missing_ollama_models_reports_unreachable_server_and_missing_models(monkeypatch) -> None:
    monkeypatch.setattr(
        rca_demo,
        "ollama_models",
        lambda url, names: {"models": {"nomic-embed-text": "abc", "gemma3:12b": None}},
    )
    assert rca_demo.missing_ollama_models("http://x", ["nomic-embed-text", "gemma3:12b"]) == [
        "gemma3:12b"
    ]

    monkeypatch.setattr(
        rca_demo, "ollama_models", lambda url, names: {"error": "refused", "models": {}}
    )
    assert rca_demo.missing_ollama_models("http://x", ["gemma3:12b"]) == "refused"
