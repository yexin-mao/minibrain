"""LlamaIndex 的公共装配：embedding、LLM、PGVectorStore。

## 为什么单独一个文件

两条 LlamaIndex 链路（`vector_rag/chain.py` 和 `graph_rag/chain.py`）
本来各自抄了一份建库逻辑，导致两个具体问题：

1. `graph_rag` 得从 `vector_rag` 里 import `_ensure_llm` / `_pg_params`
   —— **跨模块拿私有函数**，而且违反项目「模块之间不互相 import」的边界规矩
2. `search_path` 那个坑（见 `make_vector_store`）改一次要改两处

抽到这里之后，两条链路都只依赖平台层，不再互相依赖。
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import psycopg
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.postgres import PGVectorStore

from .config import get_config
from .modules.vector_rag.embeddings import embed_query, embed_texts


class MinibrainEmbedding(BaseEmbedding):
    """把项目自己的 embedding 包成 LlamaIndex 接口。

    ★★ 这个类存在的唯一目的是**锁住变量**。

    本项目对 embedding 做了 MRL 截断（4096 → 1024）+ 重新归一化。
    直接用 LlamaIndex 的 `OpenAIEmbedding` 会得到**不同的向量**，
    那样两条链路（向量 / 图）比出来的差异里就混进了「向量不一样」，
    检索结构本身的差异反而看不出来了。

    ★ 异步版本必须实现。`BaseEmbedding` 的异步批量默认会**逐条 await**，
      155 个实体就是 155 次独立 API 调用，把我们「16 条一批、4 线程并发」
      的优化整个绕过去。实测 40 条：批量 13.6s vs 串行约 100s。
      而且它**伪装成网络挂起**——进程活着、CPU 接近 0、连接数不变。
    """

    def _get_query_embedding(self, query: str) -> list[float]:
        return embed_query(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return embed_texts([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return embed_texts([text])[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)


def ensure_llm() -> None:
    """确保 `Settings.llm` 已配置。幂等。

    ★ 为什么必须有它：`QueryFusionRetriever` 即使 `num_queries=1`
      （完全不做查询改写）也会在 `__init__` 里解析 `Settings.llm`，
      没配就直接 ImportError。**框架把「查询改写」和「结果融合」
      耦合在同一个类里了**，想只用后者也得满足前者的依赖。

    ★★ 用 `OpenAILike` 而不是 `OpenAI`：后者硬编码了 OpenAI 官方模型名单，
      `deepseek/deepseek-v4-flash` 不在里面，取 `.metadata` 时直接抛
      `ValueError: Unknown model`。而且藏得很深——`num_queries=1` 时
      碰不到 `.metadata`，一开查询改写才炸。

      教训：**框架对「OpenAI 兼容」的支持程度，要看它有没有假设你就是
      OpenAI。** 这类假设通常不在文档里，只在某条冷路径上等着。
    """
    if Settings._llm is not None:
        return
    cfg = get_config()
    Settings.llm = OpenAILike(
        model=cfg.agent_model, api_base=cfg.agent_base_url,
        api_key=cfg.agent_api_key, temperature=cfg.agent_temperature,
        timeout=cfg.llm_timeout_seconds, max_retries=cfg.llm_max_retries,
        # 这两个在 OpenAI 类里是从模型名查表得来的，这里必须显式给。
        # 64k 是 deepseek 系列的保守值，只用于框架内部裁剪判断。
        context_window=65536, is_chat_model=True,
    )


def _pg_params() -> dict[str, Any]:
    url = urlparse(get_config().database_url)
    return {
        "host": url.hostname or "localhost",
        "port": url.port or 5432,
        "user": url.username or "",
        "password": url.password or "",
        "database": (url.path or "/").lstrip("/"),
    }


def make_vector_store(schema: str, table: str, *, hnsw: bool = True) -> PGVectorStore:
    """建一个 PGVectorStore，并补上框架不管的两件事。

    ★ 一、**schema 要自己建。** PGVectorStore 会建表但不建 schema。
      不先建好的话，报错指向 INSERT（`relation ... does not exist`），
      真正的原因却在更早的建库阶段——这类错最费时间。

    ★★ 二、**`search_path` 必须显式带上 `extensions`。**

      本项目**刻意**把 pgvector 扩展装在独立的 `extensions` schema，
      而不是 `public`——因为 `public` 谁都能建表，用它会破坏
      「模块够不着别人数据」这条边界（见 `db.py` 里连接池锁 search_path）。

      但 PGVectorStore 用的是数据库默认 `search_path`（`"$user", public`），
      **看不到 `extensions`**，建表时报 `type "vector" does not exist`。

      > 框架假设「扩展在 search_path 里」，而我们的架构决定把它挪走了。
      > **两个都没错，但它们不兼容——这就是换框架的真实摩擦。**

    `hnsw=False` 时不建 HNSW 索引：实体向量那种量级（几百条）
    建索引没有收益，反而多一次维护成本。
    """
    with psycopg.connect(get_config().database_url, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    kwargs: dict[str, Any] = {}
    if hnsw:
        # 参数和手写版 schema.sql 保持一致，否则两版的索引行为不可比
        kwargs["hnsw_kwargs"] = {
            "hnsw_m": 16, "hnsw_ef_construction": 64,
            "hnsw_ef_search": get_config().hnsw_ef_search,
            "hnsw_dist_method": "vector_cosine_ops",
        }

    return PGVectorStore.from_params(
        **_pg_params(), schema_name=schema, table_name=table,
        embed_dim=get_config().embedding_dimensions,
        create_engine_kwargs={
            "connect_args": {"options": f"-c search_path={schema},extensions,public"},
        },
        **kwargs,
    )


def drop_vector_table(schema: str, table: str) -> None:
    """删掉某个向量表。评测每轮开始前调，保证起点干净。

    ★ 表名是 `data_<table>` —— PGVectorStore 自己加的前缀。
    """
    with psycopg.connect(get_config().database_url, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {schema}.data_{table} CASCADE")
