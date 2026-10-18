from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DD_", env_file=".env", extra="ignore")

    dataroom_dir: Path = Path("data/dataroom")
    # "offline" = deterministic rule-based agents (no key, used in CI); "openai" = any OpenAI-compatible LLM
    llm_provider: Literal["offline", "openai"] = "offline"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    llm_timeout_s: float = 90.0
    search_k: int = 4
    max_revisions: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
