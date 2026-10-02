"""Run provenance for eval artifacts: which corpus, index, models, and code produced a result."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# Settings that change retrieval or generation results; recorded verbatim.
RESULT_SETTINGS = (
    "llm_backend",
    "generation_model",
    "agent_model",
    "embedding_model",
    "embedding_dimensions",
    "default_collection",
    "openai_chat_model",
    "openai_embed_model",
    "enable_reranker",
    "reranker_model",
    "reranker_top_k",
    "retrieval_fetch_limit",
    "retrieval_reserve_new_source",
    "retrieval_reserve_min_rerank_score",
    "chunk_size",
    "chunk_overlap",
)


def corpus_fingerprint(graph_store: Any) -> dict[str, Any]:
    """Hash of every indexed source, chunk, and edge, independent of insertion order.

    Covers node ID, title, text, and metadata (titles feed FTS5 and lexical scoring) plus
    edges (graph expansion). Uses node content rather than source_revisions because
    corpora ingested before revision tracking have no revision rows. Never raises.
    """
    try:
        with graph_store.connect() as connection:
            nodes = connection.execute(
                "SELECT id, coalesce(title, ''), coalesce(text, ''), coalesce(metadata, '') "
                "FROM nodes WHERE id LIKE 'src:%' OR id LIKE 'chk:%' ORDER BY id"
            ).fetchall()
            edges = connection.execute(
                "SELECT source, target, kind FROM edges ORDER BY source, target, kind"
            ).fetchall()
    except Exception as exc:  # e.g. "database is locked" during a concurrent ingest
        return {"error": f"{type(exc).__name__}: {exc}"}

    digest = hashlib.sha256()
    sources = chunks = 0
    for node_id, *fields in nodes:
        digest.update(node_id.encode("utf-8"))
        for field in fields:
            digest.update(b"\0")
            digest.update(hashlib.sha256(str(field).encode("utf-8")).digest())
        if node_id.startswith("src:"):
            sources += 1
        else:
            chunks += 1
    for edge in edges:
        digest.update(b"\1" + "\0".join(map(str, edge)).encode("utf-8"))
    return {
        "corpus_version": digest.hexdigest()[:16],
        "sources": sources,
        "chunks": chunks,
        "edges": len(edges),
    }


def index_snapshot(vector_store: Any) -> dict[str, Any]:
    """Vector backend state; take one at run start and one at run end to spot mid-run fallbacks."""
    collection = getattr(vector_store, "_collection", None)
    try:
        count = (
            collection.count()
            if collection is not None
            else len(getattr(vector_store, "_documents", {}))
        )
    except Exception as exc:  # a broken Chroma handle should not abort an eval run
        count = f"error: {type(exc).__name__}: {exc}"
    # read warning before backend: a store only ever falls back chroma -> json
    warning = getattr(vector_store, "backend_warning", None)
    return {"backend": getattr(vector_store, "backend", None), "records": count, "warning": warning}


def ollama_models(base_url: str, names: list[str], timeout: float = 3.0) -> dict[str, Any]:
    """Ollama server version and the exact digest of each named model (None if not installed)."""
    root = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{root}/api/version", timeout=timeout) as response:
            version = json.loads(response.read()).get("version")
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as response:
            tags = json.loads(response.read()).get("models", [])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "models": dict.fromkeys(names)}

    # name -> digest; "" means installed but no digest reported, missing key = not installed
    digests: dict[str, str] = {}
    for model in tags if isinstance(tags, list) else []:
        if isinstance(model, dict) and isinstance(model.get("name"), str):
            digests[model["name"]] = str(model.get("digest") or "")

    def digest_for(name: str) -> str | None:
        if name in digests:
            return digests[name]
        return None if ":" in name else digests.get(f"{name}:latest")

    return {"ollama_version": version, "models": {name: digest_for(name) for name in names}}


def hf_model_revision(model_name: str) -> str | None:
    """Commit hash of the locally cached Hugging Face model, if present."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    ref = Path(HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
    return ref.read_text(encoding="utf-8").strip() if ref.is_file() else None


def git_state(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
        # Pin uncommitted tracked changes so a dirty run can still be matched to its code.
        diff = (
            hashlib.sha256(git("diff", "HEAD").encode("utf-8")).hexdigest()[:16] if dirty else None
        )
        return {"commit": commit, "dirty": dirty, "diff_sha256": diff}
    except Exception as exc:
        return {
            "commit": None,
            "dirty": None,
            "diff_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_provenance(
    settings: Any,
    graph_store: Any = None,
    vector_store: Any = None,
    extra_models: list[str] | None = None,
) -> dict[str, Any]:
    """Everything needed to tell two eval runs apart from the artifact alone.

    Missing stores (e.g. test doubles) are recorded as None rather than failing the run.
    """
    wanted = [settings.generation_model, settings.agent_model, settings.embedding_model]
    names: list[str] = list(dict.fromkeys(str(name) for name in [*wanted, *(extra_models or [])]))
    models: dict[str, Any] = {"backend": settings.llm_backend}
    if settings.llm_backend == "ollama":
        models |= ollama_models(settings.llm_base_url, names)
    else:
        models["models"] = dict.fromkeys(
            [settings.openai_chat_model, settings.openai_embed_model, *(extra_models or [])]
        )
    return {
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "corpus": corpus_fingerprint(graph_store) if graph_store is not None else None,
        "index_start": index_snapshot(vector_store) if vector_store is not None else None,
        "models": models,
        "reranker": {
            "model": settings.reranker_model,
            "revision": hf_model_revision(settings.reranker_model),
        },
        "settings": {name: getattr(settings, name, None) for name in RESULT_SETTINGS},
        "code": git_state(),
        "python": platform.python_version(),
    }


def finish_provenance(provenance: dict[str, Any], vector_store: Any = None) -> dict[str, Any]:
    """Add the end-of-run index snapshot and flag a backend change during the run."""
    if vector_store is None:
        return provenance | {"index_end": None, "index_degraded_during_run": None}
    end = index_snapshot(vector_store)
    start = provenance.get("index_start") or {}
    # A change during the run, not a state the run started in (index_start shows that).
    degraded = end["backend"] != start.get("backend") or end["warning"] != start.get("warning")
    return provenance | {"index_end": end, "index_degraded_during_run": degraded}
