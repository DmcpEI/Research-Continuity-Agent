"""Model listing, ranking and tool-calling probes for the model pickers."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel

from rca.config.settings import Settings
from rca.llm.factory import get_llm_client


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


def build_model_options(
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


def build_llm_for_model(settings: Settings, model_name: str):
    return get_llm_client(
        settings.model_copy(
            update={
                "generation_model": model_name,
                "openai_chat_model": model_name,
            }
        )
    )


def fetch_available_model_names(settings: Settings, current_model: str) -> list[str]:
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


def fetch_available_model_tags(settings: Settings) -> list[dict[str, Any]]:
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


def probe_tool_calling_capability_detailed(settings: Settings, model_name: str) -> dict[str, Any]:
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
            # Unload right away: probing every installed model would otherwise hold
            # each one in RAM for Ollama's default 5 minutes.
            "keep_alive": 0,
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


def is_tool_capable_cached(
    settings: Settings,
    cache: dict[str, dict[str, Any]],
    model_name: str,
) -> bool:
    cached = cache.get(model_name)
    if isinstance(cached, dict) and "capable" in cached:
        return bool(cached.get("capable"))

    detailed = probe_tool_calling_capability_detailed(settings, model_name)
    cache[model_name] = {
        **detailed,
        "cached": False,
        "cached_at_unix": time.time(),
    }
    return bool(detailed["capable"])


def probe_and_store(
    settings: Settings,
    cache: dict[str, dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    detailed = probe_tool_calling_capability_detailed(settings, model_name)
    enriched = {
        **detailed,
        "cached": False,
        "cached_at_unix": time.time(),
    }
    cache[model_name] = enriched
    return enriched


def resolve_agent_model_selection(
    settings: Settings,
    agent: Any,
    cache: dict[str, dict[str, Any]],
    force_reprobe: bool = False,
) -> tuple[list[str], str]:
    current_model = settings.agent_model
    names = fetch_available_model_names(settings, current_model)

    capable_models: list[str] = []
    for model_name in names:
        if force_reprobe:
            probe = probe_and_store(settings, cache, model_name)
            if probe.get("capable"):
                capable_models.append(model_name)
            continue

        if is_tool_capable_cached(settings, cache, model_name):
            capable_models.append(model_name)

    if capable_models and current_model not in capable_models:
        settings.agent_model = capable_models[0]
        agent.llm = build_llm_for_model(settings, settings.agent_model)
        current_model = settings.agent_model

    return capable_models, current_model
