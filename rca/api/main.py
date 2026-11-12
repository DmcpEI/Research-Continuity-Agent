"""FastAPI application for RCA endpoints."""

from __future__ import annotations

import asyncio
import copy
import logging
import shutil
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rca.agent.loop import AgentLoop
from rca.config.settings import Settings, get_settings
from rca.contracts.nodes import NodeKind
from rca.flows.generate_flow import GeneratedAnswer, GenerateFlow
from rca.flows.ingest_flow import IngestFlow
from rca.flows.retrieve_flow import RetrieveFlow
from rca.llm.models import (
    agent_model_choices,
    build_llm_for_model,
    build_model_options,
    fetch_available_model_names,
    fetch_available_model_tags,
    probe_tool_calling_capability_detailed,
    tool_capability,
    tool_refusal_reason,
)
from rca.store.graph_store import GraphStore
from rca.store.vector_store import VectorStore, backend_status

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    model: str | None = None
    messages: list[dict[str, str]] = Field(default_factory=list)


class AgentRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    messages: list[dict[str, str]] = Field(default_factory=list)


class ModelSelectionRequest(BaseModel):
    model: str


def create_app(
    settings: Settings | None = None,
    graph_store: GraphStore | None = None,
    vector_store: VectorStore | None = None,
    retrieve_flow: RetrieveFlow | None = None,
    generate_flow: GenerateFlow | None = None,
    ingest_flow: IngestFlow | None = None,
    agent_loop: AgentLoop | None = None,
) -> FastAPI:
    """Create the RCA FastAPI application."""

    cfg = settings or get_settings()
    store = graph_store or GraphStore(cfg.graph_db_path)
    vectors = vector_store or VectorStore(cfg.vector_dir, cfg.default_collection)
    retrieve = retrieve_flow or RetrieveFlow(
        settings=cfg,
        graph_store=store,
        vector_store=vectors,
    )
    generator = generate_flow or GenerateFlow(
        settings=cfg,
        retrieve_flow=retrieve,
    )
    ingester = ingest_flow or IngestFlow(
        settings=cfg,
        graph_store=store,
        vector_store=vectors,
    )
    agent = agent_loop or AgentLoop(settings=cfg, retrieve_flow=retrieve)

    def rerankers() -> list[Any]:
        found = {}
        for flow in (retrieve, getattr(generator, "retrieve_flow", None)):
            reranker = getattr(flow, "_reranker", None)
            if reranker is not None:
                found[id(reranker)] = reranker
        return list(found.values())

    def load_rerankers() -> None:
        # At startup, so a missing model shows in /status before the first question.
        for reranker in rerankers():
            load = getattr(reranker, "load", None)
            if callable(load):
                load()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await asyncio.to_thread(load_rerankers)
        yield

    app = FastAPI(title="Research Continuity Agent API", lifespan=lifespan)
    # create_site mounts this app, and a mounted app's startup does not run; it calls this.
    app.state.load_rerankers = load_rerankers
    app.state.agent_tool_capability_cache = {}
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/sources")
    def list_sources() -> list[dict[str, Any]]:
        with store.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, kind, metadata
                FROM nodes
                WHERE kind != 'chunk'
                ORDER BY created_at DESC
                """
            ).fetchall()

        response: list[dict[str, Any]] = []
        for row in rows:
            source_id = row["id"]
            metadata = store.get_node(source_id).metadata
            latest_revision = store.find_latest_revision_for_source(source_id)
            response.append(
                {
                    "id": source_id,
                    "title": row["title"],
                    "kind": row["kind"],
                    "chunk_count": len(store.list_source_chunk_ids(source_id)),
                    "latest_path": metadata.get("latest_path"),
                    "content_sha256": metadata.get("content_sha256"),
                    "latest_revision_created_at": (
                        latest_revision.created_at.isoformat() if latest_revision else None
                    ),
                }
            )
        return response

    @router.get("/sources/{source_id:path}/pdf")
    def get_source_pdf(source_id: str) -> FileResponse:
        node = store.get_node(source_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Source not found")

        if node.kind == NodeKind.chunk:
            parent_id = str(node.metadata.get("source_id", ""))
            if not parent_id:
                raise HTTPException(status_code=404, detail="Source not found")
            parent = store.get_node(parent_id)
            if parent is None:
                raise HTTPException(status_code=404, detail="Source not found")
            node = parent

        latest_path = node.metadata.get("latest_path")
        legacy_path = node.metadata.get("path")

        raw_path: str | None = None
        is_legacy_path = False
        if isinstance(latest_path, str) and latest_path:
            raw_path = latest_path
        elif isinstance(legacy_path, str) and legacy_path:
            raw_path = legacy_path
            is_legacy_path = True

        if not raw_path:
            raise HTTPException(status_code=404, detail="PDF path not available")

        path_candidate = Path(raw_path).expanduser()
        if path_candidate.is_absolute():
            path = path_candidate.resolve()
        else:
            workspace_candidate = (cfg.workspace_root / path_candidate).expanduser().resolve()
            filesystem_candidate = (cfg.filesystem_root / path_candidate).expanduser().resolve()
            if workspace_candidate.exists():
                path = workspace_candidate
            elif filesystem_candidate.exists():
                path = filesystem_candidate
            else:
                path = workspace_candidate

        if not is_legacy_path:
            root = cfg.filesystem_root.resolve()
            if not path.is_relative_to(root):
                raise HTTPException(status_code=403, detail="Path outside allowed root")

        if path.suffix.lower() != ".pdf" or not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="PDF file not found on disk")

        return FileResponse(
            path=path,
            media_type="application/pdf",
            filename=path.name,
            content_disposition_type="inline",
        )

    @router.get("/sources/{source_id:path}")
    def get_source(source_id: str) -> dict[str, Any]:
        node = store.get_node(source_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Source not found")

        revisions = [
            revision.model_dump(mode="json") for revision in store.list_source_revisions(node.id)
        ]

        return {
            "id": node.id,
            "title": node.title,
            "kind": node.kind.value,
            "text": node.text,
            "metadata": node.metadata,
            "created_at": node.created_at.isoformat(),
            "chunk_count": len(store.list_source_chunk_ids(node.id)),
            "revisions": revisions,
        }

    @router.delete("/sources/{source_id:path}")
    def delete_source(source_id: str) -> dict[str, Any]:
        node = store.get_node(source_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Source not found")

        latest_path = node.metadata.get("latest_path")
        if isinstance(latest_path, str) and latest_path:
            path_candidate = Path(latest_path).expanduser().resolve()
            root = cfg.filesystem_root.resolve()
            if path_candidate.is_relative_to(root) and path_candidate.exists():
                path_candidate.unlink(missing_ok=True)

        chunk_ids = store.delete_source(source_id)
        if chunk_ids:
            vectors.delete_texts(chunk_ids)

        return {"deleted": source_id}

    @router.post("/chat")
    def chat(payload: ChatRequest) -> dict[str, Any]:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")

        default_model = (
            cfg.openai_chat_model
            if cfg.llm_backend == "openai_compatible"
            else cfg.generation_model
        )
        model = payload.model or default_model
        flow = generator
        if payload.model and payload.model != default_model:
            # Request-scoped: a copy with its own client, so neither the shared flow nor the
            # default model changes, and an overlapping request keeps its own model.
            flow = copy.copy(generator)
            flow.llm = build_llm_for_model(cfg, payload.model)
            if getattr(generator, "rewrite_llm", None) is generator.llm:
                flow.rewrite_llm = flow.llm
        try:
            result = flow.generate_answer(
                payload.query,
                conversation_messages=payload.messages,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Generation failed for model {model}: {_exception_message(exc)}",
            ) from exc
        return {
            "answer": result.answer,
            "citations": [citation.model_dump(mode="json") for citation in result.citations],
            "outcome": result.outcome.value,
            "grounded": result.grounded,
            "trace": result.trace.model_dump(mode="json") if result.trace else None,
            "failure_labels": result.failure_labels,
            "active_source_ids": _active_source_ids(result),
            "model": result.trace.model if result.trace and result.trace.model else model,
        }

    @router.post("/agent")
    def agent_chat(payload: AgentRequest) -> dict[str, Any]:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")

        cache: dict[str, dict[str, Any]] = app.state.agent_tool_capability_cache
        # Never switch models behind the user's back: say what is wrong and let them pick.
        if tool_capability(cfg, cache, cfg.agent_model) is False:
            reason = tool_refusal_reason(cfg, cache, cfg.agent_model)
            raise HTTPException(
                status_code=409,
                detail=(
                    f"The agent model {cfg.agent_model} cannot call tools"
                    f"{f' (Ollama: {reason})' if reason else ''}. "
                    "Pick another model in the model picker."
                ),
            )

        result = agent.run(payload.query)
        return {
            "answer": result.answer,
            "trace": result.trace.model_dump(mode="json"),
            "error": result.error,
            "model": result.trace.model,
        }

    @router.get("/models")
    def list_models() -> dict[str, Any]:
        current_model = (
            cfg.openai_chat_model
            if cfg.llm_backend == "openai_compatible"
            else cfg.generation_model
        )
        names = fetch_available_model_names(cfg, current_model)
        options = build_model_options(cfg, names, current_model)
        return {
            "models": [option.name for option in options],
            "current": current_model,
            "recommended": [
                option.model_dump(mode="json") for option in options if option.recommended
            ],
            "other": [
                option.model_dump(mode="json") for option in options if not option.recommended
            ],
        }

    @router.post("/models/select")
    def select_model(payload: ModelSelectionRequest) -> dict[str, Any]:
        selected = payload.model.strip()
        if not selected:
            raise HTTPException(status_code=400, detail="Model must not be empty")

        previous = (
            cfg.openai_chat_model
            if cfg.llm_backend == "openai_compatible"
            else cfg.generation_model
        )

        if cfg.llm_backend == "openai_compatible":
            cfg.openai_chat_model = selected
        else:
            cfg.generation_model = selected
        # Rewrites follow the answer model unless a separate rewrite model was set up;
        # otherwise the startup model stays loaded beside the new one.
        rewrites_follow = getattr(generator, "rewrite_llm", None) is generator.llm
        generator.llm = build_llm_for_model(cfg, selected)
        if rewrites_follow:
            generator.rewrite_llm = generator.llm

        current = (
            cfg.openai_chat_model
            if cfg.llm_backend == "openai_compatible"
            else cfg.generation_model
        )
        _log_ollama_processes(cfg, f"models/select {previous} -> {current}")
        return {"model": current}

    @router.get("/agent/models")
    def list_agent_models(refresh: bool = False) -> dict[str, Any]:
        cache: dict[str, dict[str, Any]] = app.state.agent_tool_capability_cache
        if refresh:
            cache.clear()
        return {
            "models": agent_model_choices(cfg),
            "current": cfg.agent_model,
            "current_capable": tool_capability(cfg, cache, cfg.agent_model),
        }

    @router.get("/agent/models/diagnostics")
    def list_agent_models_diagnostics() -> dict[str, Any]:
        # Probes every installed model on request; the pickers never do this.
        tag_map: dict[str, dict[str, Any]] = {}
        for entry in fetch_available_model_tags(cfg):
            name = entry.get("name")
            if isinstance(name, str):
                tag_map[name] = {
                    "id": entry.get("digest") or entry.get("model") or entry.get("id"),
                    "size": entry.get("size"),
                    "modified_at": entry.get("modified_at"),
                }

        records: list[dict[str, Any]] = []
        for name in agent_model_choices(cfg):
            probe = probe_tool_calling_capability_detailed(cfg, name)
            records.append(
                {
                    "name": name,
                    "present_in_ollama_tags": name in tag_map,
                    "ollama_tag": tag_map.get(name, {}),
                    "capable": bool(probe.get("capable")),
                    "probed_tag": probe.get("probed_tag"),
                    "http_status": probe.get("http_status"),
                    "error_message": probe.get("error_message", ""),
                    "error_body": probe.get("error_body", ""),
                    "cached": False,
                    "probe_path": probe.get("probe_path", "tool_probe"),
                }
            )

        return {
            "backend": cfg.llm_backend,
            "current_model": cfg.agent_model,
            "tool_capable_models": [record["name"] for record in records if record["capable"]],
            "records": records,
        }

    @router.post("/agent/models/reprobe")
    def reprobe_agent_models() -> dict[str, Any]:
        cache: dict[str, dict[str, Any]] = app.state.agent_tool_capability_cache
        cache.clear()
        return {
            "current": cfg.agent_model,
            "current_capable": tool_capability(cfg, cache, cfg.agent_model),
            "models": agent_model_choices(cfg),
            "reprobed": [cfg.agent_model],
        }

    @router.post("/agent/models/select")
    def select_agent_model(payload: ModelSelectionRequest) -> dict[str, Any]:
        selected = payload.model.strip()
        if not selected:
            raise HTTPException(status_code=400, detail="Model must not be empty")

        previous = cfg.agent_model

        cache: dict[str, dict[str, Any]] = app.state.agent_tool_capability_cache
        if tool_capability(cfg, cache, selected) is False:
            raise HTTPException(
                status_code=400,
                detail="Selected model is not tool-calling capable for agent mode",
            )

        cfg.agent_model = selected
        agent.llm = build_llm_for_model(cfg, selected)

        _log_ollama_processes(cfg, f"agent/models/select {previous} -> {cfg.agent_model}")

        return {"model": cfg.agent_model}

    @router.get("/agent/status")
    def agent_status() -> dict[str, Any]:
        cache: dict[str, dict[str, Any]] = app.state.agent_tool_capability_cache
        return {
            "model": cfg.agent_model,
            # None: Ollama could not be asked right now.
            "tool_calling_capable": tool_capability(cfg, cache, cfg.agent_model),
            "backend": cfg.llm_backend,
            "ollama_connected": _is_ollama_connected(cfg),
        }

    @router.post("/sources/ingest")
    def ingest_source(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

        original_name = Path(file.filename).name
        suffix = Path(original_name).suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(status_code=400, detail="Only .pdf files are supported")

        upload_dir = cfg.filesystem_root / ".rca" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(original_name).stem
        candidate = upload_dir / f"{stem}{suffix}"
        counter = 1
        while candidate.exists():
            candidate = upload_dir / f"{stem}-{counter}{suffix}"
            counter += 1

        with candidate.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        result = ingester.ingest_path(candidate)
        return result.model_dump(mode="json")

    @router.get("/status")
    def status() -> dict[str, Any]:
        with store.connect() as connection:
            papers = int(
                connection.execute("SELECT COUNT(*) FROM nodes WHERE kind = 'paper'").fetchone()[0]
            )
            chunks = int(
                connection.execute("SELECT COUNT(*) FROM nodes WHERE kind = 'chunk'").fetchone()[0]
            )

        model = cfg.generation_model
        if cfg.llm_backend == "openai_compatible":
            model = cfg.openai_chat_model

        # Describe the stores the flows actually use; injected flows may hold their own.
        vector_backend, warnings = backend_status(
            [
                vectors,
                getattr(retrieve, "vector_store", None),
                getattr(getattr(generator, "retrieve_flow", None), "vector_store", None),
                getattr(ingester, "vector_store", None),
            ]
        )

        warnings += [
            f"Cross-encoder reranker unavailable: {reranker.error.splitlines()[0].rstrip('.')}. "
            "Results are not reranked "
            "and the no-match gate is off."
            for reranker in rerankers()
            if getattr(reranker, "error", None)
        ]

        return {
            "papers": papers,
            "chunks": chunks,
            "backend": cfg.llm_backend,
            "model": model,
            "ollama_connected": _is_ollama_connected(cfg),
            "vector_backend": vector_backend,
            "warnings": warnings,
        }

    app.include_router(router)
    app.add_event_handler("shutdown", agent.close)
    return app


def _active_source_ids(generated: GeneratedAnswer) -> list[str]:
    source_ids = [_to_source_id(citation.source_id) for citation in generated.citations]
    if generated.trace is not None:
        source_ids.extend(_to_source_id(node_id) for node_id in generated.trace.context_node_ids)
        source_ids.extend(_to_source_id(hit.node_id) for hit in generated.trace.provenance[:8])
    return list(dict.fromkeys(source_id for source_id in source_ids if source_id))


def _to_source_id(node_id: str) -> str:
    if node_id.startswith("chk:"):
        base = node_id.rsplit(":", 1)[0]
        return "src:" + base.split(":", 1)[1]
    return node_id


def _exception_message(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return f"HTTP {exc.code} {exc.reason}: {body}".strip()
    return str(exc)


def _is_ollama_connected(settings: Settings) -> bool:
    if settings.llm_backend != "ollama":
        return False

    url = settings.llm_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _log_ollama_processes(settings: Settings, context: str) -> None:
    if settings.llm_backend != "ollama":
        return

    url = settings.llm_base_url.rstrip("/") + "/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            payload = response.read().decode("utf-8")
        logger.info("ollama /api/ps after %s: %s", context, payload)
    except Exception as exc:
        logger.warning("ollama /api/ps failed after %s: %s", context, exc)


app = create_app()


def create_site(api: FastAPI | None = None, frontend_dist: Path | None = None) -> FastAPI:
    """Serve the API under /api and the built React app at /, as the Vite dev proxy does.

    For single-process deployments such as the demo. Unknown paths return index.html so
    client-side routes (/library, /agent, ...) survive a reload.
    """
    dist = (frontend_dist or get_settings().frontend_dist).resolve()
    if not (dist / "index.html").is_file():
        raise FileNotFoundError(f"frontend build not found at {dist}; run `npm run build`")
    target = api or app

    @asynccontextmanager
    async def lifespan(_site: FastAPI):
        load = getattr(target.state, "load_rerankers", None)
        if callable(load):
            await asyncio.to_thread(load)
        yield

    site = FastAPI(
        title="Research Continuity Agent", docs_url=None, openapi_url=None, lifespan=lifespan
    )
    site.mount("/api", target)

    @site.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        try:
            candidate = (dist / path).resolve()
        except (OSError, ValueError):  # e.g. an embedded null byte
            raise HTTPException(status_code=404, detail="Not found") from None
        if candidate.is_file() and candidate.is_relative_to(dist):
            return FileResponse(candidate)
        if Path(path).suffix:  # a missing asset, not a client-side route
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(dist / "index.html")

    return site
