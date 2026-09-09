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


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    if raw.casefold() in {"1", "true", "yes", "on"}:
        return True
    if raw.casefold() in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{key} 必须是 true/false")


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
    agent_evidence_token_budget: int
    agent_evidence_limit: int
    agent_context_candidate_pool: int

    # ★★ 出站 HTTP 超时。**踩过一次才加的，别删。**
    #
    # 两处手写 OpenAI 客户端（agent/loop、embeddings）原本都没设超时。
    # OpenAI SDK 默认 600 秒 + 2 次重试，单次最坏 30 分钟；
    # 一次 194 调用的消融实验因此**挂死了 13 小时**——
    # 进程活着、CPU 只用了 14.7 秒，全程在等一个永远不返回的响应。
    #
    # 教训：`except Exception` 只挡得住「调用失败」，挡不住「调用不返回」。
    # 这是两种完全不同的故障，超时是后者唯一的防线。
    #
    # 两个值分开，因为两类调用的正常耗时差一个量级：
    #   对话类（agent）：单次问答，60 秒还不回基本就是挂了
    #   embedding：一次要批量编码几百个片段，给宽一些
    llm_timeout_seconds: float
    embedding_timeout_seconds: float
    # SDK 自带重试。设 1 而不是默认的 2：重试会把超时时间翻倍，
    # 而本项目的调用都不是幂等关键路径，快速失败比慢慢重试更有用。
    llm_max_retries: int

    # 本地 cross-encoder 重排用的模型。只有装了 rerank-ce extra 才会被加载。
    # ★ 默认 bge-reranker-base 而不是框架默认的 cross-encoder/stsb-distilroberta-base：
    #   后者是**英文 STS（句子相似度）模型**，既不是相关性重排器、也不支持中文。
    #   该模型权重本身没见过中文，不是提示词措辞的问题。
    rerank_ce_model: str

    chunk_size: int
    chunk_overlap: int
    parent_chunk_size: int
    expand_parent_context: bool

    # HNSW 查询时的候选集大小。大 → 召回高、延迟高。pgvector 默认 40。
    # 这是**近似检索**唯一的运行时旋钮，扫它就能画出「召回-延迟」权衡曲线。
    hnsw_ef_search: int

    table_query_timeout_ms: int
    table_query_max_rows: int

    ingest_max_attempts: int
    ingest_lease_seconds: int
    ingest_retry_base_seconds: int
    ingest_poll_seconds: float
    agent_run_stale_seconds: int
    upload_max_bytes: int

    session_ttl_hours: int
    agent_history_max_turns: int
    agent_history_token_budget: int

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
        agent_evidence_token_budget=_env_int("AGENT_EVIDENCE_TOKEN_BUDGET", 4000),
        agent_evidence_limit=_env_int("AGENT_EVIDENCE_LIMIT", 12),
        agent_context_candidate_pool=_env_int("AGENT_CONTEXT_CANDIDATE_POOL", 12),
        llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "60")),
        embedding_timeout_seconds=float(
            os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "180")),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 1),
        rerank_ce_model=os.environ.get("RERANK_CE_MODEL", "BAAI/bge-reranker-base"),
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 0),
        parent_chunk_size=_env_int("PARENT_CHUNK_SIZE", 1600),
        expand_parent_context=_env_bool("EXPAND_PARENT_CONTEXT", False),
        hnsw_ef_search=_env_int("HNSW_EF_SEARCH", 40),
        table_query_timeout_ms=_env_int("TABLE_QUERY_TIMEOUT_MS", 5000),
        table_query_max_rows=_env_int("TABLE_QUERY_MAX_ROWS", 200),
        ingest_max_attempts=_env_int("INGEST_MAX_ATTEMPTS", 3),
        ingest_lease_seconds=_env_int("INGEST_LEASE_SECONDS", 3600),
        ingest_retry_base_seconds=_env_int("INGEST_RETRY_BASE_SECONDS", 5),
        ingest_poll_seconds=float(os.environ.get("INGEST_POLL_SECONDS", "1")),
        agent_run_stale_seconds=_env_int("AGENT_RUN_STALE_SECONDS", 900),
        upload_max_bytes=_env_int("UPLOAD_MAX_BYTES", 20 * 1024 * 1024),
        session_ttl_hours=_env_int("SESSION_TTL_HOURS", 168),
        agent_history_max_turns=_env_int("AGENT_HISTORY_MAX_TURNS", 6),
        agent_history_token_budget=_env_int("AGENT_HISTORY_TOKEN_BUDGET", 8000),
    )

    if cfg.chunk_overlap >= cfg.chunk_size:
        raise RuntimeError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
    if cfg.parent_chunk_size < cfg.chunk_size:
        raise RuntimeError("PARENT_CHUNK_SIZE 不能小于 CHUNK_SIZE")
    # ★ 超时设成 0 或负数等于「永不超时」，那正是挂死 13 小时的那个状态。
    #   这里 fail-fast，不允许把防线关掉。
    if cfg.llm_timeout_seconds <= 0 or cfg.embedding_timeout_seconds <= 0:
        raise RuntimeError("LLM_TIMEOUT_SECONDS / EMBEDDING_TIMEOUT_SECONDS 必须为正数")
    if cfg.embedding_dimensions <= 0:
        raise RuntimeError("EMBEDDING_DIMENSIONS 必须为正整数")
    if (cfg.agent_evidence_token_budget <= 0 or cfg.agent_evidence_limit <= 0
            or cfg.agent_context_candidate_pool <= 0):
        raise RuntimeError("AGENT_EVIDENCE_* / AGENT_CONTEXT_CANDIDATE_POOL 必须为正数")
    if (cfg.ingest_max_attempts <= 0 or cfg.ingest_lease_seconds <= 0
            or cfg.ingest_retry_base_seconds <= 0 or cfg.ingest_poll_seconds <= 0
            or cfg.agent_run_stale_seconds <= 0):
        raise RuntimeError("INGEST_* 任务参数必须为正数")
    if cfg.upload_max_bytes <= 0:
        raise RuntimeError("UPLOAD_MAX_BYTES 必须为正整数")
    if cfg.agent_history_max_turns <= 0 or cfg.agent_history_token_budget <= 0:
        raise RuntimeError("AGENT_HISTORY_MAX_TURNS / TOKEN_BUDGET 必须为正整数")

    _config = cfg
    return cfg
