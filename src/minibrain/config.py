"""环境变量配置。启动时一次性读取并校验，不在业务代码里散读 os.environ。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """极简 .env 加载：不覆盖已存在的真实环境变量。"""
    for parent in [Path.cwd(), *Path(__file__).resolve().parents]:
        env_file = parent / ".env"
        if not env_file.is_file():
            continue
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
        return


def _env(key: str, default: str | None = None) -> str:
    value = os.environ.get(key, default)
    if value is None or value == "":
        raise RuntimeError(f"缺少环境变量 {key}，参考 .env.example")
    return value


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return default if raw is None or raw == "" else int(raw)


@dataclass(frozen=True)
class Config:
    database_url: str

    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimensions: int

    agent_base_url: str
    agent_api_key: str
    agent_model: str
    agent_temperature: float
    agent_max_steps: int

    chunk_size: int
    chunk_overlap: int

    table_query_timeout_ms: int
    table_query_max_rows: int

    session_ttl_hours: int

    @property
    def agent_configured(self) -> bool:
        return not self.agent_api_key.startswith("<")

    @property
    def embedding_configured(self) -> bool:
        return not self.embedding_api_key.startswith("<")


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is not None:
        return _config

    _load_dotenv()
    cfg = Config(
        database_url=_env("DATABASE_URL"),
        embedding_base_url=_env("EMBEDDING_BASE_URL"),
        embedding_api_key=_env("EMBEDDING_API_KEY"),
        embedding_model=_env("EMBEDDING_MODEL"),
        embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", 1024),
        agent_base_url=_env("AGENT_BASE_URL"),
        agent_api_key=_env("AGENT_API_KEY"),
        agent_model=_env("AGENT_MODEL"),
        agent_temperature=float(os.environ.get("AGENT_TEMPERATURE", "0")),
        agent_max_steps=_env_int("AGENT_MAX_STEPS", 6),
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
        table_query_timeout_ms=_env_int("TABLE_QUERY_TIMEOUT_MS", 5000),
        table_query_max_rows=_env_int("TABLE_QUERY_MAX_ROWS", 200),
        session_ttl_hours=_env_int("SESSION_TTL_HOURS", 168),
    )

    if cfg.chunk_overlap >= cfg.chunk_size:
        raise RuntimeError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
    if cfg.embedding_dimensions <= 0:
        raise RuntimeError("EMBEDDING_DIMENSIONS 必须为正整数")

    _config = cfg
    return cfg
