"""Runtime settings loaded from environment variables."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseModel

    BaseSettings = BaseModel
    _HAS_PYDANTIC_SETTINGS = False


class Settings(BaseSettings):
    """Application settings with conservative local defaults."""

    app_name: str = "research-continuity-agent"
    environment: str = "development"
    workspace_root: Path = Field(default_factory=Path.cwd)
    filesystem_root: Path = Field(default_factory=Path.cwd)
    enable_filesystem_tools: bool = True
    data_dir: Path = Field(default_factory=lambda: Path(".rca"))
    graph_db_path: Path = Field(default_factory=lambda: Path(".rca/graph.sqlite3"))
    vector_dir: Path = Field(default_factory=lambda: Path(".rca/vectors"))
    event_log_path: Path = Field(default_factory=lambda: Path(".rca/events.jsonl"))
    telemetry_log_path: Path = Field(default_factory=lambda: Path(".rca/telemetry.jsonl"))
    experiment_db_path: Path = Field(default_factory=lambda: Path(".rca/experiments.sqlite3"))
    default_collection: str = "research-continuity"
    tool_policy_path: Path = Field(default_factory=lambda: Path("rca/config/tool_policies.yaml"))
    chunk_size: int = 1200
    chunk_overlap: int = 150
    embedding_dimensions: int = 768
    llm_backend: Literal["ollama", "openai_compatible"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434"  # Ollama default
    llm_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("RCA_LLM_BASE_URL", "LLM_BASE_URL"),
    )
    llm_api_key: str = Field(
        default="ollama",
        validation_alias=AliasChoices("RCA_LLM_API_KEY", "LLM_API_KEY"),
    )
    generation_model: str = "qwen2.5:14b"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("RCA_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10
    retrieval_fetch_limit: int = 20

    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="RCA_",
            env_file=".env",
            extra="ignore",
            populate_by_name=True,
        )
    else:
        model_config = {"extra": "ignore", "populate_by_name": True}

    def __init__(self, **data: object) -> None:
        if _HAS_PYDANTIC_SETTINGS:
            super().__init__(**data)
            return

        merged_data = self._load_dotenv_file(Path(".env"))
        for env_name, field_name in self._env_field_map().items():
            if env_name in os.environ:
                merged_data[field_name] = os.environ[env_name]
        merged_data.update(data)
        super().__init__(**merged_data)

    def ensure_runtime_directories(self) -> None:
        """Create local runtime directories when they do not exist."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.graph_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_db_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _env_field_map() -> dict[str, str]:
        return {
            "RCA_ENVIRONMENT": "environment",
            "RCA_WORKSPACE_ROOT": "workspace_root",
            "RCA_FILESYSTEM_ROOT": "filesystem_root",
            "RCA_ENABLE_FILESYSTEM_TOOLS": "enable_filesystem_tools",
            "RCA_DATA_DIR": "data_dir",
            "RCA_GRAPH_DB_PATH": "graph_db_path",
            "RCA_VECTOR_DIR": "vector_dir",
            "RCA_EVENT_LOG_PATH": "event_log_path",
            "RCA_TELEMETRY_LOG_PATH": "telemetry_log_path",
            "RCA_EXPERIMENT_DB_PATH": "experiment_db_path",
            "RCA_DEFAULT_COLLECTION": "default_collection",
            "RCA_TOOL_POLICY_PATH": "tool_policy_path",
            "RCA_CHUNK_SIZE": "chunk_size",
            "RCA_CHUNK_OVERLAP": "chunk_overlap",
            "RCA_EMBEDDING_DIMENSIONS": "embedding_dimensions",
            "RCA_LLM_BACKEND": "llm_backend",
            "RCA_EMBEDDING_MODEL": "embedding_model",
            "RCA_EMBEDDING_BASE_URL": "embedding_base_url",
            "RCA_LLM_BASE_URL": "llm_base_url",
            "LLM_BASE_URL": "llm_base_url",
            "RCA_LLM_API_KEY": "llm_api_key",
            "LLM_API_KEY": "llm_api_key",
            "RCA_GENERATION_MODEL": "generation_model",
            "RCA_OPENAI_BASE_URL": "openai_base_url",
            "RCA_OPENAI_API_KEY": "openai_api_key",
            "OPENAI_API_KEY": "openai_api_key",
            "RCA_OPENAI_CHAT_MODEL": "openai_chat_model",
            "RCA_OPENAI_EMBED_MODEL": "openai_embed_model",
            "RCA_ENABLE_RERANKER": "enable_reranker",
            "RCA_RERANKER_MODEL": "reranker_model",
            "RCA_RERANKER_TOP_K": "reranker_top_k",
            "RCA_RETRIEVAL_FETCH_LIMIT": "retrieval_fetch_limit",
        }

    @staticmethod
    def _load_dotenv_file(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}

        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", maxsplit=1)
            values[Settings._env_field_map().get(key, key)] = raw_value.strip().strip("\"'")
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    settings = Settings()
    settings.ensure_runtime_directories()
    return settings
