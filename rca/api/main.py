"""FastAPI application for RCA endpoints."""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import urllib.error
import urllib.request
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
from rca.llm.factory import get_llm_client
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


class ModelOption(BaseModel):
    name: str
    recommended: bool
    score: int
    reasons: list[str]


_EMBEDDING_PATTERNS = (
    "embed",
    "embedding",
    "bge",
    "e5",
    "gte",
    "jina-embeddings",
    "nomic-embed",
)
_VISION_PATTERNS = ("vision", "vl", "llava", "minicpm-v", "internvl")
_TEXT_FAMILY_PATTERNS = (
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "phi",
    "deepseek",
    "mixtral",
    "yi",
    "nemotron",
)
_CODING_SPECIALIZED_PATTERNS = ("coder", "code", "starcoder", "codestral")


def _split_tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def _contains_any(name: str, patterns: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(pattern in lowered for pattern in patterns)


def _rank_model(name: str, preferred: list[str]) -> tuple[int, list[str]]:
    lowered = name.lower()
    reasons: list[str] = []
    score = 0

    for index, preferred_name in enumerate(preferred):
        if lowered == preferred_name.lower():
            boost = max(50 - index * 10, 20)
            score += boost
            reasons.append("preferred")
            break

    if _contains_any(name, _TEXT_FAMILY_PATTERNS):
        score += 30
        reasons.append("strong text family")

    if any(tag in lowered for tag in ("instruct", "chat")):
        score += 15
        reasons.append("instruction tuned")

    if _contains_any(name, _VISION_PATTERNS):
        score -= 18
        reasons.append("multimodal")

    if _contains_any(name, _CODING_SPECIALIZED_PATTERNS):
        score -= 20
        reasons.append("code oriented")

    if ":" in name and "b" in lowered:
        size_match = re.search(r"(\d+(?:\.\d+)?)b", lowered)
        if size_match is not None:
            try:
                size_b = float(size_match.group(1))
                if size_b < 3:
                    score -= 12
                    reasons.append("small capacity")
                elif 6 <= size_b <= 20:
                    score += 8
                    reasons.append("good local size")
            except ValueError:
                pass

    return score, reasons


def _build_model_options(
    settings: Settings, model_names: list[str], current: str
) -> list[ModelOption]:
    include_set = {value.lower() for value in settings.model_recommend_include}
    exclude_set = {value.lower() for value in settings.model_recommend_exclude}
    preferred = settings.model_recommend_preferred

    deduped: list[str] = []
    seen: set[str] = set()
    for name in model_names:
        normalized = name.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)

    if current and current.lower() not in seen:
        deduped.insert(0, current)

    options: list[ModelOption] = []
    for name in deduped:
        lowered = name.lower()
        if lowered in exclude_set:
            continue

        forced_include = lowered in include_set or lowered == current.lower()
        if not forced_include and _contains_any(name, _EMBEDDING_PATTERNS):
            continue

        score, reasons = _rank_model(name, preferred)
        recommended = forced_include or score >= 10
        options.append(
            ModelOption(
                name=name,
                recommended=recommended,
                score=score,
                reasons=reasons,
            )
        )

    options.sort(key=lambda option: (-int(option.recommended), -option.score, option.name.lower()))
    return options


def _build_llm_for_model(settings: Settings, model_name: str):
    return get_llm_client(
        settings.model_copy(
            update={
                "generation_model": model_name,
                "openai_chat_model": model_name,
            }
        )
    )


def _fetch_available_model_names(settings: Settings, current_model: str) -> list[str]:
    names: list[str] = []
    if settings.llm_backend != "ollama":
        return [current_model] if current_model else []

    tags_url = settings.llm_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=2.0) as response:
            payload = response.read().decode("utf-8")
            parsed = json.loads(payload)
            models = parsed.get("models", [])
            if isinstance(models, list):
                for model in models:
                    if isinstance(model, dict) and isinstance(model.get("name"), str):
                        names.append(model["name"])
    except Exception:
        names = []

    if current_model and current_model not in names:
        names.insert(0, current_model)
    return names


def _fetch_available_model_tags(settings: Settings) -> list[dict[str, Any]]:
    if settings.llm_backend != "ollama":
        return []

    tags_url = settings.llm_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=2.0) as response:
            payload = response.read().decode("utf-8")
            parsed = json.loads(payload)
            models = parsed.get("models", [])
            if isinstance(models, list):
                return [model for model in models if isinstance(model, dict)]
    except Exception:
        return []

    return []


def _tag_aliases(model_name: str) -> list[str]:
    aliases = [model_name]
    if ":" not in model_name:
        aliases.append(f"{model_name}:latest")
    return aliases


def _probe_tool_calling_capability(settings: Settings, model_name: str) -> bool:
    return bool(_probe_tool_calling_capability_detailed(settings, model_name)["capable"])


def _probe_tool_calling_capability_detailed(settings: Settings, model_name: str) -> dict[str, Any]:
    if settings.llm_backend != "ollama":
        return {
            "capable": True,
            "probed_tag": model_name,
            "http_status": 200,
            "error_body": "",
            "error_message": "",
            "probe_path": "skipped_non_ollama",
        }

    last_error: dict[str, Any] = {
        "capable": False,
        "probed_tag": model_name,
        "http_status": None,
        "error_body": "",
        "error_message": "probe_failed_without_detail",
        "probe_path": "tool_probe",
    }

    for tag in _tag_aliases(model_name):
        payload = {
            "model": tag,
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "No-op tool capability probe",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                    },
                }
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1},
        }

        for timeout_seconds in (5.0, 60.0):
            request = urllib.request.Request(
                settings.llm_base_url.rstrip("/") + "/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds):
                    return {
                        "capable": True,
                        "probed_tag": tag,
                        "http_status": 200,
                        "error_body": "",
                        "error_message": "",
                        "probe_path": "tool_probe",
                    }
            except urllib.error.HTTPError as exc:
                raw_body = ""
                try:
                    raw_body = exc.read().decode("utf-8")
                except Exception:
                    raw_body = ""

                last_error = {
                    "capable": False,
                    "probed_tag": tag,
                    "http_status": exc.code,
                    "error_body": raw_body,
                    "error_message": str(exc),
                    "probe_path": "tool_probe",
                }
                break
            except TimeoutError as exc:
                last_error = {
                    "capable": False,
                    "probed_tag": tag,
                    "http_status": None,
                    "error_body": "",
                    "error_message": str(exc) or "timed out",
                    "probe_path": f"tool_probe_timeout_{int(timeout_seconds)}s",
                }
                # Retry once with a slower timeout for large models.
                continue
            except Exception as exc:
                last_error = {
                    "capable": False,
                    "probed_tag": tag,
                    "http_status": None,
                    "error_body": "",
                    "error_message": str(exc),
                    "probe_path": "tool_probe",
                }
                break

    return last_error


def _is_tool_capable_cached(
    settings: Settings,
    cache: dict[str, dict[str, Any]],
    model_name: str,
) -> bool:
    cached = cache.get(model_name)
    if isinstance(cached, dict) and "capable" in cached:
        return bool(cached.get("capable"))

    detailed = _probe_tool_calling_capability_detailed(settings, model_name)
    cache[model_name] = {
        **detailed,
        "cached": False,
        "cached_at_unix": time.time(),
    }
    return bool(detailed["capable"])


def _probe_and_store(
    settings: Settings,
    cache: dict[str, dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    detailed = _probe_tool_calling_capability_detailed(settings, model_name)
    enriched = {
        **detailed,
        "cached": False,
        "cached_at_unix": time.time(),
    }
    cache[model_name] = enriched
    return enriched


def _resolve_agent_model_selection(
    settings: Settings,
    agent: AgentLoop,
    cache: dict[str, dict[str, Any]],
    force_reprobe: bool = False,
) -> tuple[list[str], str]:
    current_model = settings.agent_model
    names = _fetch_available_model_names(settings, current_model)

    capable_models: list[str] = []
    for model_name in names:
        if force_reprobe:
            probe = _probe_and_store(settings, cache, model_name)
            if probe.get("capable"):
                capable_models.append(model_name)
            continue

        if _is_tool_capable_cached(settings, cache, model_name):
            capable_models.append(model_name)

    if capable_models and current_model not in capable_models:
        settings.agent_model = capable_models[0]
        agent.llm = _build_llm_for_model(settings, settings.agent_model)
        current_model = settings.agent_model

    return capable_models, current_model


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

    app = FastAPI(title="Research Continuity Agent API")
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

        if payload.model:
            if cfg.llm_backend == "openai_compatible":
                cfg.openai_chat_model = payload.model
            else:
                cfg.generation_model = payload.model
            generator.llm = _build_llm_for_model(cfg, payload.model)

        model = payload.model or (
            cfg.openai_chat_model
            if cfg.llm_backend == "openai_compatible"
            else cfg.generation_model
        )
        try:
            result = generator.generate_answer(
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
        _resolve_agent_model_selection(cfg, agent, cache)

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
        names = _fetch_available_model_names(cfg, current_model)
        options = _build_model_options(cfg, names, current_model)
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
        generator.llm = _build_llm_for_model(cfg, selected)

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
        capable_models, current_model = _resolve_agent_model_selection(
            cfg,
            agent,
            cache,
            force_reprobe=refresh,
        )

        return {
            "models": capable_models,
            "current": current_model,
        }

    @router.get("/agent/models/diagnostics")
    def list_agent_models_diagnostics(refresh: bool = False) -> dict[str, Any]:
        cache: dict[str, dict[str, Any]] = app.state.agent_tool_capability_cache
        capable_models, current_model = _resolve_agent_model_selection(
            cfg,
            agent,
            cache,
            force_reprobe=refresh,
        )

        tag_entries = _fetch_available_model_tags(cfg)
        tag_map: dict[str, dict[str, Any]] = {}
        for entry in tag_entries:
            name = entry.get("name")
            if isinstance(name, str):
                tag_map[name] = {
                    "id": entry.get("digest") or entry.get("model") or entry.get("id"),
                    "size": entry.get("size"),
                    "modified_at": entry.get("modified_at"),
                }

        names = _fetch_available_model_names(cfg, cfg.agent_model)
        records: list[dict[str, Any]] = []
        for name in names:
            cached = cache.get(name)
            if cached is None:
                cached = _probe_and_store(cfg, cache, name)

            records.append(
                {
                    "name": name,
                    "present_in_ollama_tags": name in tag_map,
                    "ollama_tag": tag_map.get(name, {}),
                    "capable": bool(cached.get("capable")),
                    "probed_tag": cached.get("probed_tag"),
                    "http_status": cached.get("http_status"),
                    "error_message": cached.get("error_message", ""),
                    "error_body": cached.get("error_body", ""),
                    "cached": True,
                    "cached_at_unix": cached.get("cached_at_unix"),
                    "probe_path": cached.get("probe_path", "tool_probe"),
                }
            )

        return {
            "backend": cfg.llm_backend,
            "current_model": current_model,
            "tool_capable_models": capable_models,
            "records": records,
        }

    @router.post("/agent/models/reprobe")
    def reprobe_agent_models() -> dict[str, Any]:
        cache: dict[str, dict[str, Any]] = app.state.agent_tool_capability_cache
        cache.clear()
        capable_models, current_model = _resolve_agent_model_selection(
            cfg,
            agent,
            cache,
            force_reprobe=True,
        )

        return {
            "current": current_model,
            "models": capable_models,
            "reprobed": list(cache.keys()),
        }

    @router.post("/agent/models/select")
    def select_agent_model(payload: ModelSelectionRequest) -> dict[str, Any]:
        selected = payload.model.strip()
        if not selected:
            raise HTTPException(status_code=400, detail="Model must not be empty")

        previous = cfg.agent_model

        cache: dict[str, dict[str, Any]] = app.state.agent_tool_capability_cache
        if not _is_tool_capable_cached(cfg, cache, selected):
            raise HTTPException(
                status_code=400,
                detail="Selected model is not tool-calling capable for agent mode",
            )

        cfg.agent_model = selected
        agent.llm = _build_llm_for_model(cfg, selected)

        _log_ollama_processes(cfg, f"agent/models/select {previous} -> {cfg.agent_model}")

        return {"model": cfg.agent_model}

    @router.get("/agent/status")
    def agent_status() -> dict[str, Any]:
        cache: dict[str, dict[str, Any]] = app.state.agent_tool_capability_cache
        capable_models, current_model = _resolve_agent_model_selection(cfg, agent, cache)
        capability = current_model in capable_models if capable_models else False

        return {
            "model": current_model,
            "tool_calling_capable": capability,
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
