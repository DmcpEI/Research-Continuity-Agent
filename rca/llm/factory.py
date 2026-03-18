"""Centralized LLM client construction."""

from __future__ import annotations

from rca.config.settings import Settings, get_settings
from rca.llm.client import EchoLLMClient, LLMClient, OllamaLLMClient


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build the configured chat + embedding client for the active backend."""

    settings = settings or get_settings()

    if not settings.generation_model:
        return EchoLLMClient()

    if settings.llm_backend == "openai_compatible":
        return OllamaLLMClient(
            base_url=settings.openai_base_url,
            model=settings.openai_chat_model,
            embedding_model=settings.openai_embed_model,
            api_key=settings.openai_api_key,
            api_style="openai",
        )

    return OllamaLLMClient(
        base_url=settings.llm_base_url,
        model=settings.generation_model,
        embedding_model=settings.embedding_model,
        api_key=settings.llm_api_key,
        api_style="ollama",
    )
