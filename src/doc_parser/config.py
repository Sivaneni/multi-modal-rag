"""Configuration management using pydantic-settings."""
from __future__ import annotations

import logging

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Parser backend
    parser_backend: str = "cloud"  # "cloud" | "ollama"
    z_ai_api_key: SecretStr | None = None
    log_level: str = "INFO"
    output_dir: str = "./output"
    config_yaml_path: str = "config.yaml"

    # OpenAI
    openai_api_key: SecretStr | None = None
    openai_llm_model: str = "gpt-4o"

    # Embedding (provider-agnostic)
    embedding_provider: str = "openai"  # "openai" | "gemini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    gemini_api_key: SecretStr | None = None

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "documents"

    # LLM provider for generation / captioning / graph extraction
    llm_provider: str = "meshapi"  # "meshapi" | "openai"
    mesh_api_key: SecretStr | None = None

    # Neo4j (optional — graph extraction is skipped when neo4j_uri is not set)
    neo4j_uri: str | None = None
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr | None = None

    # Reranker
    reranker_backend: str = "openai"  # "jina" | "openai" | "bge" | "qwen"
    reranker_top_n: int = 5
    jina_api_key: SecretStr | None = None

    # Feature flags
    image_caption_enabled: bool = True

    # Captioning tuning
    table_max_tokens: int = 2000
    table_max_input_chars: int = 12_000
    image_max_tokens: int = 800
    table_use_vision: bool = False

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Logging
    log_json: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        **kwargs: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # .env file takes priority over system environment variables
        return init_settings, dotenv_settings, env_settings

    @model_validator(mode="after")
    def _validate_backend(self) -> Settings:
        """Enforce backend-specific constraints and auto-set config path."""
        if self.parser_backend == "cloud":
            if self.z_ai_api_key is None:
                raise ValueError(
                    "Z_AI_API_KEY is required when PARSER_BACKEND=cloud"
                )
        elif self.parser_backend == "ollama":
            if self.config_yaml_path == "config.yaml":
                self.config_yaml_path = "ollama/config.yaml"
        else:
            raise ValueError(
                f"PARSER_BACKEND must be 'cloud' or 'ollama', got: {self.parser_backend!r}"
            )
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


_MESHAPI_BASE_URL = "https://api.meshapi.ai/v1"


def make_async_llm_client() -> "AsyncOpenAI":  # type: ignore[name-defined]
    """Return an AsyncOpenAI client pointed at the configured LLM provider."""
    from openai import AsyncOpenAI
    settings = get_settings()
    if settings.llm_provider == "meshapi" and settings.mesh_api_key:
        return AsyncOpenAI(
            api_key=settings.mesh_api_key.get_secret_value(),
            base_url=_MESHAPI_BASE_URL,
        )
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return AsyncOpenAI(api_key=api_key)


def make_sync_llm_client() -> "OpenAI":  # type: ignore[name-defined]
    """Return a sync OpenAI client pointed at the configured LLM provider."""
    from openai import OpenAI
    settings = get_settings()
    if settings.llm_provider == "meshapi" and settings.mesh_api_key:
        return OpenAI(
            api_key=settings.mesh_api_key.get_secret_value(),
            base_url=_MESHAPI_BASE_URL,
        )
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return OpenAI(api_key=api_key)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with the given level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
