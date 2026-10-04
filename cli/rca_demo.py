"""Public demo: fetch the sample corpus, ingest it, and serve the React app plus API on one port."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from rca.config.settings import get_settings
from rca.telemetry.provenance import ollama_models

MANIFEST = Path(__file__).resolve().parents[1] / "demo" / "papers.json"
USER_AGENT = "research-continuity-agent-demo"


def missing_ollama_models(base_url: str, names: list[str]) -> list[str] | str:
    """Models not installed on the Ollama server, or an error string if it is unreachable."""
    found = ollama_models(base_url, names)
    if "error" in found:
        return found["error"]
    return [name for name, digest in found["models"].items() if digest is None]


def download_missing(papers: list[dict[str, str]], target: Path) -> list[Path]:
    """Download each paper's pinned arXiv version unless it is already on disk."""
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for paper in papers:
        path = target / Path(paper["file"]).name
        if not path.is_file():
            print(f"downloading {paper['arxiv']} {paper['title'][:60]}", flush=True)
            request = urllib.request.Request(
                f"https://arxiv.org/pdf/{paper['arxiv']}", headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
            if not body.startswith(b"%PDF"):
                raise RuntimeError(f"arXiv did not return a PDF for {paper['arxiv']}")
            partial = path.with_suffix(".part")
            partial.write_bytes(body)
            partial.replace(path)  # never leave a truncated PDF under the final name
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.llm_backend == "ollama":
        names = [settings.embedding_model, settings.generation_model]
        # Chat and embeddings both go through llm_base_url (rca/llm/factory.py).
        missing = missing_ollama_models(settings.llm_base_url, names)
        if isinstance(missing, str):
            print(
                f"Ollama is not reachable at {settings.llm_base_url} ({missing}).\n"
                "Start it on the host (`ollama serve`) and run the demo again.",
                file=sys.stderr,
            )
            return 1
        if missing:
            pulls = "\n".join(f"  ollama pull {name}" for name in missing)
            print(f"Missing Ollama models; on the host run:\n{pulls}", file=sys.stderr)
            return 1

    papers = json.loads(MANIFEST.read_text(encoding="utf-8"))["papers"]
    try:
        paths = download_missing(papers, settings.data_dir / "papers")
    except (OSError, RuntimeError) as exc:  # URLError and HTTP 429/503 are OSErrors
        print(
            f"Could not download the sample papers from arXiv: {exc}\n"
            "Check the network and run the demo again; finished files are kept.",
            file=sys.stderr,
        )
        return 1

    from rca.flows.ingest_flow import IngestFlow

    ingest = IngestFlow(settings=settings)
    for path in paths:  # re-ingest is idempotent: unchanged files skip all writes
        result = ingest.ingest_path(path)
        print(f"{result.ingest_status:9s} {result.source_id}", flush=True)
    if ingest.vector_store.backend != "chroma":
        print(f"warning: {ingest.vector_store.backend_warning}", file=sys.stderr)

    import uvicorn

    from rca.api.main import create_site

    print(f"RCA demo on http://localhost:{args.port}", flush=True)
    uvicorn.run(create_site(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
