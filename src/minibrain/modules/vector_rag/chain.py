"""LlamaIndex 版检索链路。和手写的 `core.py` **并行存在**，用于对比。

## 为什么做这一版

手写版把 RAG 的每个环节都自己实现了一遍（切分 / BM25 / RRF / 检索）。
那让人清楚每一步在干什么，但**岗位要求会框架**，而且只有把同一件事
用两种方式做出来，才谈得上"我知道框架帮我做了什么、代价是什么"。

这一版刻意做成**逐环节对位**，方便一条条比：

| 环节 | 手写版 | LlamaIndex 版 |
|---|---|---|
| 切分 | `chunking.split_text`（按段落聚合） | `SentenceSplitter` |
| 向量库 | 自己写 SQL + pgvector | `PGVectorStore` |
| 关键词 | `keyword.py` 手写 BM25 + 倒排索引 | `BM25Retriever` |
| 融合 | `fusion.reciprocal_rank_fusion` | `QueryFusionRetriever`（RRF 模式） |
| 权限 | SQL WHERE 里的可见性条件 | `MetadataFilters` |

## ★★ 必须锁住的变量：embedding

两版**用同一个 embedding 函数**，字节级一致。

原因是本项目对 embedding 做了 MRL 截断（4096 → 1024）+ 重新归一化。
直接用 LlamaIndex 的 `OpenAIEmbedding` 会得到**不同的向量**，
那样比出来的差异里就混进了"向量不一样"这个变量，
检索/切分/融合的差异反而看不出来了。

所以下面把项目自己的 `embed_texts` / `embed_query` 包成 LlamaIndex 的
`BaseEmbedding`。**换框架时必须锁住不参与比较的变量**，这条比结论重要。

同理，BM25 的分词器也复用 `keyword.tokenize`——否则中文会被
LlamaIndex 默认的空白分词切成一坨，比的就成了分词而不是检索。

## 权限：框架能不能守住这条线

本项目的硬约束是「可见性判定永远写在 SQL 的 WHERE 里，不做查完再筛」。
应用层后过滤是最容易长越权 bug 的地方。

`MetadataFilters` 会被 `PGVectorStore` 翻译成 SQL 的 WHERE 条件（作用在
metadata 的 JSONB 列上），所以**理论上**能满足。但"理论上能"不算数——
`tests/test_chain.py` 里有一条测试实际验证
「别人的私有文档检索不到」。框架换了，这条线一寸都不能退。

## 已知代价（不是缺点，是取舍）

★ `BM25Retriever` 把**全部节点放在内存里**算。手写版为此专门建了
  Postgres 倒排索引（探针十一，快 87.6 倍）。
  这一版在语料大了之后会退化——框架的默认实现不解决这个问题。
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.core.vector_stores import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
)
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LIOpenAI
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.postgres import PGVectorStore

from ...config import get_config
from ...contracts import Evidence, ModuleError, SearchResult, UserContext
from .embeddings import embed_query, embed_texts
from .tokenizer import tokenize

MODULE_ID = "vector-rag"

# 独立 schema，和手写版的 mod_vector 物理隔开。
# ★ 两版共用一份语料但各存各的，比较时不会互相污染；
#   也顺带证明"模块边界"这条规矩在换框架后仍然成立。
SCHEMA = "mod_vector_li"
TABLE = "nodes"

_store: PGVectorStore | None = None
_index: VectorStoreIndex | None = None


class MinibrainEmbedding(BaseEmbedding):
    """把项目自己的 embedding 包成 LlamaIndex 接口。

    ★★ 这个类存在的唯一目的是**锁住变量**：两版必须用完全相同的向量，
      否则比出来的差异说明不了任何问题。见模块 docstring。
    """

    def _get_query_embedding(self, query: str) -> list[float]:
        return embed_query(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return embed_texts([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        # 复用项目自己的批量 + 线程池实现，连并发行为都一致
        return embed_texts(texts)


def _pg_params() -> dict[str, Any]:
    url = urlparse(get_config().database_url)
    return {
        "host": url.hostname or "localhost",
        "port": url.port or 5432,
        "user": url.username or "",
        "password": url.password or "",
        "database": (url.path or "/").lstrip("/"),
    }


def _ensure_schema() -> None:
    """PGVectorStore 会建表，但**不会建 schema**——不先建好就 UndefinedTable。

    这类"框架替你做了一半"的地方，是换框架时最容易卡住的那种问题：
    报错信息指向 INSERT，真正的原因却在更早的建库阶段。
    """
    import psycopg
    with psycopg.connect(get_config().database_url, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")


def get_store() -> PGVectorStore:
    """★★ 这个函数里有两处「框架撞上本项目架构」的适配，都不是可选的。

    **一、schema 要自己建。** PGVectorStore 会建表，但不建 schema。
    报错信息指向 INSERT（`relation ... does not exist`），
    真正的原因在更早的建库阶段——这类错最费时间。

    **二、`search_path` 要显式带上 `extensions`。** 这条更值得记：

        建表时报 `type "vector" does not exist`

    本项目**刻意**把 pgvector 扩展装在独立的 `extensions` schema，
    而不是 `public`——因为 `public` 谁都能建表，用它会破坏
    「模块够不着别人数据」这条边界（见 db.py 里连接池锁 search_path 那段）。

    但 PGVectorStore 用的是数据库默认 `search_path`（`"$user", public`），
    **看不到 `extensions`**，于是 `vector` 类型不存在。

    > 框架假设「扩展在 search_path 里」，而我们的架构决定把它挪走了。
    > **两个都没错，但它们不兼容——这就是换框架的真实摩擦。**

    手写版没这个问题，因为 `db.py` 的连接池本来就把 search_path
    锁成 `<schema>, extensions`。
    """
    global _store
    if _store is None:
        _ensure_schema()
        _store = PGVectorStore.from_params(
            **_pg_params(),
            schema_name=SCHEMA,
            table_name=TABLE,
            embed_dim=get_config().embedding_dimensions,
            # HNSW 参数和手写版 schema.sql 里保持一致，否则索引行为不可比
            hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64,
                         "hnsw_ef_search": get_config().hnsw_ef_search,
                         "hnsw_dist_method": "vector_cosine_ops"},
            create_engine_kwargs={
                "connect_args": {
                    "options": f"-c search_path={SCHEMA},extensions,public"},
            },
        )
    return _store


def _ensure_llm() -> None:
    """★ QueryFusionRetriever 即使 num_queries=1 也**强制**要一个 LLM。

    它在 __init__ 里就去解析 `Settings.llm`，没配就直接 ImportError。
    可我们只想要它的 RRF 融合，完全不需要查询改写——
    **框架把「查询改写」和「结果融合」耦合在同一个类里了**，
    想只用后者就得连前者的依赖一起满足。

    手写版的 `fusion.reciprocal_rank_fusion` 是 104 行纯函数，不依赖任何模型。
    这就是「框架帮你做了什么、代价是什么」里的代价那一半。
    """
    if Settings._llm is None:
        cfg = get_config()
        Settings.llm = LIOpenAI(
            model=cfg.agent_model, api_base=cfg.agent_base_url,
            api_key=cfg.agent_api_key, temperature=cfg.agent_temperature,
            timeout=cfg.llm_timeout_seconds, max_retries=cfg.llm_max_retries)


def get_index() -> VectorStoreIndex:
    global _index
    if _index is None:
        _index = VectorStoreIndex.from_vector_store(
            get_store(), embed_model=MinibrainEmbedding())
    return _index


def ingest(user: UserContext, filename: str, text: str,
           *, source_name: str | None = None,
           visibility: str = "private") -> int:
    """把一篇文档切分、向量化、写进 LlamaIndex 的库。返回节点数。

    ★ metadata 里必须带 owner_id 和 visibility —— 检索时的权限过滤全靠它们。
      手写版是把可见性写进 SQL 的 WHERE；这一版是写进 metadata 再让
      MetadataFilters 翻译成 WHERE。**落点不同，但都在 SQL 层，没有后过滤。**
    """
    cfg = get_config()
    splitter = SentenceSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        # ★ SentenceSplitter 默认按 token 数切，而本项目的 CHUNK_SIZE 是**字符数**
        #   （切分消融测出来 800 字符最优）。不改成字符口径的话，
        #   两版切出来的粒度差好几倍，比的就不是同一件事了。
        tokenizer=list,
    )
    doc = Document(
        text=text,
        metadata={
            "filename": filename,
            "source_name": source_name or f"user/{user.username}",
            "owner_id": str(user.user_id),
            "visibility": visibility,
        },
    )
    nodes = splitter.get_nodes_from_documents([doc])
    for i, node in enumerate(nodes):
        node.metadata["ordinal"] = i
    get_index().insert_nodes(nodes)
    return len(nodes)


def ingest_raw(*, filename: str, text: str, source_name: str,
               visibility: str, owner_id: str) -> int:
    """后台入库入口。不收 UserContext —— 权限在登记阶段已经判过了。

    ★ 和 `ingest()` 的区别只是参数来源：这个版本从 documents 表里读元信息，
      给 `core.process` 调用。分开是因为后台任务不代表任何用户发起新的访问，
      这条规矩和 `gateway.process` 不带 UserContext 是同一个道理。
    """
    cfg = get_config()
    splitter = SentenceSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
        tokenizer=list,
    )
    doc = Document(text=text, metadata={
        "filename": filename, "source_name": source_name,
        "owner_id": owner_id, "visibility": visibility,
    })
    nodes = splitter.get_nodes_from_documents([doc])
    for i, node in enumerate(nodes):
        node.metadata["ordinal"] = i
    get_index().insert_nodes(nodes)
    return len(nodes)


def _visibility_filters(user: UserContext) -> MetadataFilters | None:
    """和手写版 `_visibility_clause` 一一对应的过滤条件。

    管理员看全部（返回 None = 不加过滤）；
    普通用户看「public 的」或「自己的」——**OR 关系**，
    和手写版那句 `(s.visibility = 'public' OR s.owner_id = %s)` 同构。
    """
    if user.is_admin:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(key="visibility", value="public",
                           operator=FilterOperator.EQ),
            MetadataFilter(key="owner_id", value=str(user.user_id),
                           operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR,
    )


def search(user: UserContext, query: str, top_k: int = 5,
           mode: str = "hybrid") -> SearchResult:
    """和手写版同签名同返回，方便同一套评测脚本两版都能跑。

    ★★ 开头这两句校验是**重构时丢过一次的**，值得记：

      手写版 `search()` 第一句就是「空查询直接报错」。换成 LlamaIndex 之后
      我没把它带过来，于是空查询一路走到 BM25，打出一片零分，
      日志里只留下一行 `The query is empty` —— **不报错，只是返回垃圾**。

      是 `test_empty_query_is_rejected` 把它抓出来的。

    > **框架负责算法，不负责你的输入契约。** 换执行引擎时，
    > 那些「一句话的守卫」最容易被漏掉，因为它们不在框架的抽象里，
    > 也不会因为缺失而报错。
    """
    query = query.strip()
    if not query:
        raise ModuleError("查询不能为空", code="empty_query")
    if mode not in ("hybrid", "vector", "keyword"):
        raise ModuleError(f"未知检索模式 {mode}", code="unknown_search_mode")

    filters = _visibility_filters(user)
    vector_retriever = VectorIndexRetriever(
        index=get_index(), similarity_top_k=top_k, filters=filters)

    if mode == "vector":
        nodes = vector_retriever.retrieve(query)
    else:
        # ★ BM25Retriever 需要节点在内存里。手写版为此建了 Postgres 倒排索引
        #   （探针十一，快 87.6 倍）——框架的默认实现不解决这个问题。
        #   这里先用向量路捞一个候选池，再在池内做 BM25，
        #   是对内存问题的妥协；语料大了要换别的方案。
        pool = VectorIndexRetriever(
            index=get_index(), similarity_top_k=top_k * 8, filters=filters
        ).retrieve(query)
        if not pool:
            return SearchResult(evidence=[], note="没有可检索的内容。")
        bm25 = BM25Retriever.from_defaults(
            nodes=[n.node for n in pool],
            similarity_top_k=top_k,
            # 分词器复用手写版，否则中文会被默认的空白分词毁掉，
            # 比的就成了分词质量而不是检索链路
            tokenizer=tokenize,
        )
        _ensure_llm()
        fusion = QueryFusionRetriever(
            [vector_retriever, bm25],
            similarity_top_k=top_k,
            num_queries=1,                 # 不做查询改写：手写版没有，要对齐
            mode="reciprocal_rerank",      # = RRF，和 fusion.py 同一个算法
            use_async=False,
        )
        nodes = fusion.retrieve(query)

    if not nodes:
        return SearchResult(evidence=[], note="没有命中任何内容。")

    return SearchResult(evidence=[
        Evidence(
            module=MODULE_ID,
            source_name=n.node.metadata.get("source_name", "?"),
            location=f"{n.node.metadata.get('filename', '?')} "
                     f"#{n.node.metadata.get('ordinal', 0)}",
            snippet=n.node.get_content(),
            score=round(float(n.score), 4) if n.score is not None else None,
        )
        for n in nodes[:top_k]
    ])


def drop_all() -> None:
    """清空这一版的存储。评测脚本每轮开始前调，保证起点干净。"""
    global _store, _index
    import psycopg

    with psycopg.connect(get_config().database_url, autocommit=True) as conn:
        conn.execute(f'DROP TABLE IF EXISTS {SCHEMA}.data_{TABLE} CASCADE')
    _store, _index = None, None
