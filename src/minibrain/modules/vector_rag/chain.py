"""LlamaIndex 版检索链路。和手写的 `core.py` **并行存在**，用于对比。

## 为什么做这一版

手写版把 RAG 的每个环节都自己实现了一遍（切分 / BM25 / RRF / 检索）。
那让人清楚每一步在干什么，但**岗位要求会框架**，而且只有把同一件事
用两种方式做出来，才谈得上"我知道框架帮我做了什么、代价是什么"。

这一版刻意做成**逐环节对位**，方便一条条比：

| 环节 | 手写版 | LlamaIndex 版 |
|---|---|---|
| 切分 | `chunking.split_text`（按段落聚合） | Markdown 章节 + `SentenceSplitter` 父子切分 |
| 向量库 | 自己写 SQL + pgvector | `PGVectorStore` |
| 关键词 | `keyword.py` 手写 BM25 + 倒排索引 | `lexical.py` 持久化全库 BM25 |
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

同理，BM25 的分词器也复用 `tokenizer.py`——否则中文会被
默认的空白分词切成一坨，比的就成了分词而不是检索。

## 权限：框架能不能守住这条线

本项目的硬约束是「可见性判定永远写在 SQL 的 WHERE 里，不做查完再筛」。
应用层后过滤是最容易长越权 bug 的地方。

`MetadataFilters` 会被 `PGVectorStore` 翻译成 SQL 的 WHERE 条件（作用在
metadata 的 JSONB 列上），所以**理论上**能满足。但"理论上能"不算数——
`tests/test_chain.py` 里有一条测试实际验证
「别人的私有文档检索不到」。框架换了，这条线一寸都不能退。

## 已知代价（不是缺点，是取舍）

★ 框架默认 `BM25Retriever` 需要把全部节点放在内存里。主路径没有接受
  这个扩展性代价，而是在 `mod_vector_li` 维护倒排索引，让 sparse 路从
  全库独立召回。中文仍只索引标识符，这是现有消融数据得出的取舍。
"""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Callable

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
)
from llama_index.core import Settings, VectorStoreIndex

from ...config import get_config
from ...contracts import (
    Evidence, ModuleError, RetrievalCandidate, RetrievalStage, RetrievalTrace,
    SearchResult, UserContext,
)
from ...db import vector_index_db
from ...llamaindex_setup import (
    MinibrainEmbedding, drop_vector_table, ensure_llm, make_vector_store,
)
from .metadata import build_filters, extract_document_metadata, extract_query_metadata
from .hierarchy import ParentContext, build_hierarchy
from . import lexical
from .selection import deduplicate_nodes, mmr_select
from .confidence import extract_retrieval_signals
from .index_manifest import clear_manifest, ensure_index_compatible

MODULE_ID = "vector-rag"

# 独立 schema，和手写版的 mod_vector 物理隔开。
# ★ 两版共用一份语料但各存各的，比较时不会互相污染；
#   也顺带证明"模块边界"这条规矩在换框架后仍然成立。
SCHEMA = "mod_vector_li"
TABLE = "nodes"

_index: VectorStoreIndex | None = None
_manifest_validated = False


def get_index(*, validate_manifest: bool = True) -> VectorStoreIndex:
    global _index, _manifest_validated
    if validate_manifest and not _manifest_validated:
        ensure_index_compatible()
        _manifest_validated = True
    if _index is None:
        _index = VectorStoreIndex.from_vector_store(
            make_vector_store(SCHEMA, TABLE), embed_model=MinibrainEmbedding())
    return _index


def _prepare_nodes(documents: list[tuple[str, str]], *, source_name: str,
                   visibility: str, owner_id: str,
                   document_ids: list[str | None] | None = None):
    """一次切分多篇文档，供普通入库和公开评测批量入库共用。"""
    nodes, _ = _prepare_hierarchy(
        documents, source_name=source_name, visibility=visibility,
        owner_id=owner_id, document_ids=document_ids)
    return nodes


def _prepare_hierarchy(documents: list[tuple[str, str]], *, source_name: str,
                       visibility: str, owner_id: str,
                       document_ids: list[str | None] | None = None
                       ) -> tuple[list, list[ParentContext]]:
    """按 Markdown 章节构造 child 检索节点和 parent 上下文。"""
    cfg = get_config()
    nodes = []
    parents: list[ParentContext] = []
    ids = document_ids or [None] * len(documents)
    if len(ids) != len(documents):
        raise ValueError("document_ids 必须和 documents 等长")
    for (filename, text), document_id in zip(documents, ids):
        document_nodes, document_parents = build_hierarchy(
            filename=filename, text=text, source_name=source_name,
            visibility=visibility, owner_id=owner_id, document_id=document_id,
            child_size=cfg.chunk_size, child_overlap=cfg.chunk_overlap,
            parent_size=cfg.parent_chunk_size,
            document_metadata=extract_document_metadata(filename, text),
        )
        nodes.extend(document_nodes)
        parents.extend(document_parents)
    return nodes, parents


def _store_parent_contexts(parents: list[ParentContext]) -> None:
    if not parents:
        return
    with vector_index_db() as cur:
        cur.executemany(
            """INSERT INTO parent_contexts
                 (id, document_id, source_name, owner_id, visibility, filename,
                  ordinal, heading_path, content, content_hash)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                 content = EXCLUDED.content, heading_path = EXCLUDED.heading_path,
                 content_hash = EXCLUDED.content_hash""",
            [(p.id, p.document_id, p.source_name, p.owner_id, p.visibility,
              p.filename, p.ordinal, p.heading_path, p.content, p.content_hash)
             for p in parents],
        )


def _delete_parent_ids(parent_ids: list[str]) -> None:
    if parent_ids:
        with vector_index_db() as cur:
            cur.execute("DELETE FROM parent_contexts WHERE id = ANY(%s)", (parent_ids,))


def ingest_many_raw(documents: list[tuple[str, str]], *, source_name: str,
                    visibility: str, owner_id: str,
                    document_ids: list[str | None] | None = None) -> int:
    """批量入库；公开检索评测用它避免 5000 次逐文档 API/SQL 往返。"""
    if not documents:
        return 0
    nodes, parents = _prepare_hierarchy(
        documents, source_name=source_name, visibility=visibility,
        owner_id=owner_id, document_ids=document_ids)
    parent_ids = [parent.id for parent in parents]
    _store_parent_contexts(parents)
    try:
        get_index().insert_nodes(nodes)
        try:
            lexical.index_nodes(nodes)
        except Exception:
            # sparse 与 dense 要么都成功，要么都不留下本批的半成品。
            get_index(validate_manifest=False).vector_store.delete_nodes(
                node_ids=[node.node_id for node in nodes])
            raise
    except Exception:
        _delete_parent_ids(parent_ids)
        raise
    return len(nodes)


def ingest(user: UserContext, filename: str, text: str,
           *, source_name: str | None = None,
           visibility: str = "private") -> int:
    """把一篇文档切分、向量化、写进 LlamaIndex 的库。返回节点数。

    ★ metadata 里必须带 owner_id 和 visibility —— 检索时的权限过滤全靠它们。
      手写版是把可见性写进 SQL 的 WHERE；这一版是写进 metadata 再让
      MetadataFilters 翻译成 WHERE。**落点不同，但都在 SQL 层，没有后过滤。**
    """
    return ingest_many_raw(
        [(filename, text)], source_name=source_name or f"user/{user.username}",
        visibility=visibility, owner_id=str(user.user_id))


def ingest_raw(*, filename: str, text: str, source_name: str,
               visibility: str, owner_id: str, document_id: str) -> int:
    """后台入库入口。不收 UserContext —— 权限在登记阶段已经判过了。

    ★ 和 `ingest()` 的区别只是参数来源：这个版本从 documents 表里读元信息，
      给 `core.process` 调用。分开是因为后台任务不代表任何用户发起新的访问，
      这条规矩和 `gateway.process` 不带 UserContext 是同一个道理。
    """
    return ingest_many_raw(
        [(filename, text)], source_name=source_name,
        visibility=visibility, owner_id=owner_id,
        document_ids=[document_id])


def delete_source_nodes(*, owner_id: str, source_name: str) -> None:
    """删除一个 source 在 LlamaIndex 表中的节点，避免注册表删了但向量残留。"""
    filters = MetadataFilters(filters=[
        MetadataFilter(key="owner_id", value=owner_id, operator=FilterOperator.EQ),
        MetadataFilter(key="source_name", value=source_name, operator=FilterOperator.EQ),
    ], condition=FilterCondition.AND)
    get_index(validate_manifest=False).vector_store.delete_nodes(filters=filters)
    lexical.delete_source(owner_id=owner_id, source_name=source_name)
    with vector_index_db() as cur:
        cur.execute("DELETE FROM parent_contexts WHERE owner_id = %s AND source_name = %s",
                    (owner_id, source_name))


def delete_document_nodes(*, owner_id: str, document_id: str) -> None:
    """按稳定 registry_document_id 同时删除 dense 与 sparse 节点。"""
    filters = MetadataFilters(filters=[
        MetadataFilter(key="owner_id", value=owner_id, operator=FilterOperator.EQ),
        MetadataFilter(key="registry_document_id", value=document_id,
                       operator=FilterOperator.EQ),
    ], condition=FilterCondition.AND)
    get_index(validate_manifest=False).vector_store.delete_nodes(filters=filters)
    lexical.delete_document(owner_id=owner_id, document_id=document_id)
    with vector_index_db() as cur:
        cur.execute("DELETE FROM parent_contexts WHERE owner_id = %s AND document_id = %s",
                    (owner_id, document_id))


def _visibility_filters(user: UserContext) -> MetadataFilters | None:
    """和手写版 `_visibility_clause` 一一对应的过滤条件。

    管理员看全部（返回 None = 不加过滤）；
    普通用户看「public 的」或「自己的」——**OR 关系**，
    和手写版那句 `(s.visibility = 'public' OR s.owner_id = %s)` 同构。
    """
    return build_filters(user, "", business=False)


def _within_document_filters(user: UserContext, filename: str) -> MetadataFilters:
    """权限条件与文件名精确过滤在 SQL 中做 AND，不在召回后筛。"""
    filename_filter = MetadataFilter(
        key="filename", value=filename, operator=FilterOperator.EQ)
    permission = _visibility_filters(user)
    if permission is None:
        return MetadataFilters(filters=[filename_filter], condition=FilterCondition.AND)
    return MetadataFilters(
        filters=[permission, filename_filter], condition=FilterCondition.AND)


def _within_source_filters(user: UserContext, source_name: str) -> MetadataFilters:
    """权限条件与知识域精确过滤一起下推到 PGVectorStore。"""
    source_filter = MetadataFilter(
        key="source_name", value=source_name, operator=FilterOperator.EQ)
    permission = _visibility_filters(user)
    if permission is None:
        return MetadataFilters(filters=[source_filter], condition=FilterCondition.AND)
    return MetadataFilters(
        filters=[permission, source_filter], condition=FilterCondition.AND)


# ★★ 查询改写用的中文提示词。**必须自己写，不能用框架默认的。**
#
# LlamaIndex 的 QUERY_GEN_PROMPT 是英文的：
#     "You are a helpful assistant that generates multiple search queries..."
# 中文语料下拿它去改写，模型很可能吐出英文查询——然后拿英文去检索中文文档，
# 召回直接塌掉。**这是"框架默认值是给英文语料调的"的典型例子**，
# 和 BM25Retriever 默认按空白分词是同一类问题。
_QUERY_GEN_PROMPT_ZH = (
    "你在为一个中文知识库生成检索用的查询。\n"
    "针对下面这个问题，生成 {num_queries} 条**中文**检索查询，每行一条。\n"
    "要求：\n"
    "- 换用不同的说法和关键词，覆盖同一个意图的不同表达\n"
    "- 保留原问题里的专有名词、编号、人名（那些是精确命中的关键）\n"
    "- 不要解释，不要编号，只输出查询本身\n\n"
    "问题：{query}\n"
    "查询：\n"
)


def _rewrite_with_hyde(query: str) -> str:
    """HyDE：先让模型**编一个假答案**，拿假答案去检索。

    直觉是这样的：问题和答案的用词往往不一样。
    「住宿能报多少」和「差旅住宿标准：一线城市每晚不超过 600 元」
    在向量空间里未必近，但一个**编造的答案**和真答案用词接近得多。

    ★ 编出来的内容可能完全是错的——**这不重要**。
      HyDE 用的是它的"词面形态"，不是它的事实性。检索完之后
      假答案就被丢掉，进 prompt 的仍然是真实召回的片段。

    ★ 代价：多一次 LLM 调用（实测约 2~5 秒），而且假答案可能把检索带偏
      ——问题越冷门，模型编得越离谱，偏得越远。值不值要看消融数据。
    """
    ensure_llm()
    transform = HyDEQueryTransform(llm=Settings.llm, include_original=True)
    bundle = transform.run(query)
    # include_original=True 时 embedding_strs 是 [假答案, 原问题]，
    # 拼在一起送进检索——保留原问题是为了不让编造内容完全主导语义。
    return "\n".join(bundle.embedding_strs)


def _fetch_k(top_k: int, rerank_pool: int, candidate_pool: int = 0) -> int:
    """开了重排就多召回一些 —— **截断推迟到重排之后**。

    ★★ 顺序是这件事唯一的收益来源：召回 fetch_k 条 → 重排 → 截 top_k。

      如果先截到 top_k 再重排，重排就只是把 5 条重新排 5 条，
      救不回原本排在第 6~20 名的必需文档 —— 而那正是 Complete Recall@5
      丢分的地方（k=5 上限 0.931，实测 0.707，空间 0.224 全在这里）。

    ★ rerank_pool 必须 > top_k 才有意义：重排只能重新排列已经召回的东西，
      **它变不出没召回到的文档**。所以 pool 比 top_k 还小时取 top_k，
      而不是让候选池反而缩水。

    提成独立函数是为了能测 —— 它埋在 search() 里的时候，
    「有没有提前截断」这条不变量没有任何测试碰得到。
    """
    return max(top_k, rerank_pool, candidate_pool)


def strip_metadata_for_rerank(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """让节点在 `MetadataMode.EMBED` 下只吐正文，不吐 metadata。

    `excluded_embed_metadata_keys` 是 LlamaIndex 控制这件事的官方开关，
    不是 hack。就地改，因为检索回来的节点是本次请求的临时对象。

    提成模块级函数是为了能测 —— 留在类里的话，测它就得先构造
    `SentenceTransformerRerank`，而那要求 sentence-transformers 装好并加载真模型。
    **一条不该依赖 2GB 依赖才能验证的不变量。**
    """
    for item in nodes:
        item.node.excluded_embed_metadata_keys = list(item.node.metadata.keys())
    return nodes


class MinibrainCrossEncoderRerank(SentenceTransformerRerank):
    """本地 cross-encoder 重排；可选增强，不依赖生成式 LLM。

    ## 为什么加它

    LLM 重排实测 **7383ms**，而不重排是 11ms —— 慢 671 倍。
    当前对重排的初步否定（「买到的是省 context，不是更准」）里，
    **这 671 倍的分母是「用 LLM 当重排器」贡献的，不是「重排」这件事本身的**。
    不把这个变量分离出来，「重排不值得做」这个结论不成立。

    cross-encoder 是业界标准做法：专用小模型，20 条候选约几十毫秒。

    ## 直接复用框架实现，但先核对它的截断和输入行为

    `SentenceTransformerRerank._postprocess_nodes` 读过了，算法上没问题：

    - `assert len(scores) == len(nodes)` —— **不丢文档**
      不会因模型输出漏编号而静默丢候选
    - 每条独立打分再整体排序 —— 不分批，分数同一把尺子
    - 每对独立过模型 —— **天然没有位置偏见**，不需要打乱那套机制
    - 根本没有提示词 —— 没有「框架默认提示词是英文」的问题

    **框架的实现好不好要一个个读，不能因为上一个有问题就都自己写。**

    ## 只接管两件事

    **一、模型默认值。** 框架默认 `cross-encoder/stsb-distilroberta-base`，
    那是**英文 STS（句子相似度）模型**，既不是相关性重排器也不支持中文。
    改成 `BAAI/bge-reranker-base`（可用 `RERANK_CE_MODEL` 覆盖）。

    **二、送进模型的文本。**

    ★★ 框架用 `get_content(MetadataMode.EMBED)`，那会把 metadata 拼在正文前面：

            filename: team-backend.md
            source_name: user/alice
            owner_id: 3f2a9c11-7b4e-4d21-9a8c-1e5f6b0d2a44
            visibility: private
            ordinal: 0

            后端组组长是张敏，共 8 人。

      一个 UUID 和一个恒为 private 的字段进重排器纯粹是噪声，而且短片段里
      噪声占比可能过半。这里用官方的 `excluded_embed_metadata_keys`
      把 metadata 全部排除，只送正文 —— **和手写版喂给重排的 `r["content"]` 对齐**，
      否则两条重排路线的输入就不是同一个东西，比出来的差异说明不了问题。

    > 同一个坑在**入库 embedding** 上也存在，而且更严重。见 `ingest()` 的注释。
    """

    @classmethod
    def class_name(cls) -> str:
        return "MinibrainCrossEncoderRerank"

    def _postprocess_nodes(self, nodes: list[NodeWithScore],
                           query_bundle: QueryBundle | None = None) -> list[NodeWithScore]:
        # 只送正文，不送 metadata。理由见类 docstring 第二条。
        return super()._postprocess_nodes(
            strip_metadata_for_rerank(nodes), query_bundle)


@lru_cache(maxsize=1)
def _load_cross_encoder_rerank() -> MinibrainCrossEncoderRerank:
    """进程内只加载一份权重；不同 top_n 共享底层 CrossEncoder。"""
    return MinibrainCrossEncoderRerank(
        top_n=1, model=get_config().rerank_ce_model)


def make_cross_encoder_rerank(top_n: int) -> MinibrainCrossEncoderRerank:
    """构造 cross-encoder 重排器。没装 extra 时给出可操作的错误，而不是 ImportError。

    ★ 依赖是**可选**的：torch + sentence-transformers 约 2GB，
      而项目其余部分全部走 API、零本地模型。不该让每个 `uv sync` 都付这个成本。
    """
    try:
        import sentence_transformers                       # noqa: F401  PLC0415
    except ImportError as exc:
        raise ModuleError(
            "cross-encoder 重排需要额外依赖，请先 `uv sync --extra rerank-ce`",
            code="rerank_ce_not_installed", status=503) from exc

    # 模型约 1.1GB，绝不能按查询或候选数重新加载。Pydantic 的浅拷贝会保留
    # 同一个 private `_model` 引用，只改变本次 postprocessor 的 top_n，
    # 因此并发请求之间也不会通过修改共享实例的 top_n 互相污染。
    return _load_cross_encoder_rerank().model_copy(
        update={"top_n": top_n}, deep=False)


# 重排器注册表。★ 消融时 `reranker` 是**唯一变量** ——
# 召回方式、候选池大小、截断时机对两条路完全相同，差异只可能来自重排器本身。
# 这和「锁住 embedding」是同一个手法：先把不参与比较的变量钉死，再谈差异。
RERANKERS: dict[str, Callable[[int], BaseNodePostprocessor]] = {
    "ce": make_cross_encoder_rerank,
}


def _keyword_nodes(user: UserContext, query: str, fetch_k: int,
                   business: dict[str, object]) -> list[NodeWithScore]:
    ranked = lexical.rank_node_ids(user, query, fetch_k, business=business)
    if not ranked:
        return []
    stored = get_index().vector_store.get_nodes(node_ids=[node_id for node_id, _ in ranked])
    by_id = {node.node_id: node for node in stored}
    return [NodeWithScore(node=by_id[node_id], score=score)
            for node_id, score in ranked if node_id in by_id]


def _candidate_snapshot(nodes: list[NodeWithScore]) -> list[RetrievalCandidate]:
    """立刻复制阶段结果，避免后续 RRF/rerank 就地改分数污染早期轨迹。"""
    result = []
    for rank, item in enumerate(nodes, start=1):
        metadata = item.node.metadata
        result.append(RetrievalCandidate(
            rank=rank,
            node_id=str(item.node.node_id),
            source_name=str(metadata.get("source_name", "?")),
            location=f"{metadata.get('filename', '?')} #{metadata.get('ordinal', 0)}",
            snippet=item.node.get_content()[:500],
            score=round(float(item.score), 6) if item.score is not None else None,
        ))
    return result


def _add_stage(stages: list[RetrievalStage] | None, name: str,
               score_kind: str, nodes: list[NodeWithScore], started: float,
               note: str = "") -> None:
    if stages is not None:
        stages.append(RetrievalStage(
            name=name, score_kind=score_kind,
            candidates=_candidate_snapshot(nodes),
            latency_ms=round((perf_counter() - started) * 1000, 2), note=note,
        ))


def _retrieve_nodes(user: UserContext, retrieval_query: str, *, mode: str,
                    num_queries: int, fetch_k: int,
                    filters: MetadataFilters | None,
                    business: dict[str, object],
                    stages: list[RetrievalStage] | None = None,
                    stage_suffix: str = "") -> list[NodeWithScore]:
    """执行一次检索。metadata 安全回退由外层控制，避免这里隐式放宽权限。"""
    if mode == "keyword":
        started = perf_counter()
        nodes = _keyword_nodes(user, retrieval_query, fetch_k, business)
        _add_stage(stages, f"BM25{stage_suffix}", "BM25", nodes, started)
        return nodes

    vector_retriever = VectorIndexRetriever(
        index=get_index(), similarity_top_k=fetch_k, filters=filters)

    started = perf_counter()
    if num_queries <= 1:
        vector_nodes = vector_retriever.retrieve(retrieval_query)
    else:
        ensure_llm()
        vector_nodes = QueryFusionRetriever(
            [vector_retriever], similarity_top_k=fetch_k,
            num_queries=num_queries, query_gen_prompt=_QUERY_GEN_PROMPT_ZH,
            mode="reciprocal_rerank", use_async=False,
        ).retrieve(retrieval_query)
    vector_note = ("单次向量召回" if num_queries <= 1
                   else f"{num_queries} 条查询在 dense 路内部做 RRF")
    _add_stage(stages, f"Dense 召回{stage_suffix}", "向量相似度", vector_nodes,
               started, vector_note)

    if mode == "vector":
        return vector_nodes

    started = perf_counter()
    keyword_nodes = _keyword_nodes(user, retrieval_query, fetch_k, business)
    _add_stage(stages, f"BM25 召回{stage_suffix}", "BM25", keyword_nodes, started,
               "从全库稀疏索引独立召回，不受 dense 候选池限制")

    # 两条路从全库独立召回后再融合；关键词不再受向量候选池截断。
    started = perf_counter()
    fused = lexical.fuse_rankings(vector_nodes, keyword_nodes, fetch_k)
    _add_stage(stages, f"RRF 融合{stage_suffix}", "RRF", fused, started,
               "只融合名次，不直接比较不同量纲的原始分数")
    return fused


def _hydrate_embeddings(nodes: list[NodeWithScore]) -> None:
    """从 PGVectorStore 读回候选已有向量，供 MMR 算候选间冗余。

    这是一次数据库读取，不会再次调用 embedding API。读不到时 selection.py 会退回
    确定性的词项 Jaccard，相比让整个检索失败更合适。
    """
    missing = [item.node.node_id for item in nodes if not item.node.embedding]
    if not missing:
        return
    try:
        stored = get_index().vector_store.get_nodes(node_ids=missing)
    except Exception:  # noqa: BLE001 - MMR 是增强层，失败时仍应返回基础检索结果
        return
    embeddings = {node.node_id: node.embedding for node in stored if node.embedding}
    for item in nodes:
        if not item.node.embedding and item.node.node_id in embeddings:
            item.node.embedding = embeddings[item.node.node_id]


def _load_visible_parents(user: UserContext, parent_ids: list[str]) -> dict[str, dict]:
    """批量读取父块；权限再次落在 SQL WHERE，不信任调用方传来的节点。"""
    if not parent_ids:
        return {}
    with vector_index_db() as cur:
        if user.is_admin:
            cur.execute("SELECT * FROM parent_contexts WHERE id = ANY(%s)",
                        (parent_ids,))
        else:
            cur.execute(
                """SELECT * FROM parent_contexts
                   WHERE id = ANY(%s) AND (visibility = 'public' OR owner_id = %s)""",
                (parent_ids, str(user.user_id)),
            )
        return {str(row["id"]): dict(row) for row in cur.fetchall()}


def _evidence_from_ranked_nodes(user: UserContext, nodes: list[NodeWithScore],
                                top_k: int, *, expand_parent: bool = True,
                                deduplicate_parent: bool = False) -> list[Evidence]:
    """把同一 child 排名投影成原片段、parent-aware 去重或完整 parent。"""
    parent_ids = list(dict.fromkeys(
        str(item.node.metadata["parent_context_id"])
        for item in nodes if expand_parent
        and item.node.metadata.get("parent_context_id")))
    parents = _load_visible_parents(user, parent_ids) if parent_ids else {}
    evidence: list[Evidence] = []
    seen: set[str] = set()

    for item in nodes:
        metadata = item.node.metadata
        parent_id = str(metadata.get("parent_context_id", ""))
        parent = parents.get(parent_id) if expand_parent else None
        key = (f"parent:{parent_id}"
               if parent_id and (parent or deduplicate_parent)
               else f"node:{item.node.node_id}")
        if key in seen:
            continue
        seen.add(key)
        if parent:
            heading = str(parent.get("heading_path") or "")
            # 保留原 Evidence location 的 ``filename #...`` 契约；P 表示 parent。
            location = f"{parent['filename']} #P{parent['ordinal']}"
            if heading:
                location += f"（{heading}）"
            snippet = str(parent["content"])
            source_name = str(parent["source_name"])
        else:
            location = f"{metadata.get('filename', '?')} #{metadata.get('ordinal', 0)}"
            heading = str(metadata.get("heading_path") or "")
            if heading:
                location += f"（{heading}）"
            snippet = item.node.get_content()
            source_name = str(metadata.get("source_name", "?"))
        evidence.append(Evidence(
            module=MODULE_ID, source_name=source_name, location=location,
            snippet=snippet,
            score=round(float(item.score), 4) if item.score is not None else None,
        ))
        if len(evidence) >= top_k:
            break
    return evidence


def search(user: UserContext, query: str, top_k: int = 5,
           mode: str = "hybrid", *,
           num_queries: int = 1, hyde: bool = False,
           rerank_pool: int = 0, reranker: str = "ce",
           candidate_pool: int = 0,
           use_mmr: bool = False, mmr_lambda: float = 0.7,
           metadata_filtering: bool = True,
           within_filename: str | None = None,
           within_source: str | None = None,
           expand_parent: bool | None = None,
           explain: bool = False) -> SearchResult:
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
    if rerank_pool > 0 and reranker not in RERANKERS:
        raise ModuleError(f"未知重排器 {reranker}", code="unknown_reranker")
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ModuleError("mmr_lambda 必须在 0 到 1 之间", code="invalid_mmr_lambda")
    if candidate_pool < 0:
        raise ModuleError("candidate_pool 不能为负数", code="invalid_candidate_pool")
    if within_filename is not None:
        within_filename = within_filename.strip()
        if not within_filename:
            raise ModuleError("filename 不能为空", code="invalid_filename")
    if within_source is not None:
        within_source = within_source.strip()
        if not within_source:
            raise ModuleError("source 不能为空", code="invalid_source")
    if within_filename and within_source:
        raise ModuleError("filename 和 source 不能同时指定", code="conflicting_scope")
    if expand_parent is None:
        expand_parent = get_config().expand_parent_context

    # ★ HyDE 在**检索前**改写查询；num_queries 是在**检索时**生成多条查询各查一遍。
    #   两者可以叠加，但那样一次问答要调 1(HyDE) + (n-1)(改写) 次 LLM，
    #   成本会翻好几倍——值不值要看消融数据，不要默认全开。
    retrieval_query = _rewrite_with_hyde(query) if hyde else query

    # MMR 必须先看到比 top-k 更大的池子，否则从 5 条里选 5 条只会改变顺序，
    # 救不回被同一来源挤到第 6 名之后的证据。
    # 一个父块通常对应约两个子块；多取一倍 child，展开去重后仍能尽量填满 top-k。
    fetch_k = max(_fetch_k(top_k, rerank_pool, candidate_pool), top_k * 2)
    if use_mmr:
        fetch_k = max(fetch_k, top_k * 4)

    business = extract_query_metadata(query) if metadata_filtering else {}
    if within_filename:
        business = {"filename": within_filename}
        filters = _within_document_filters(user, within_filename)
    elif within_source:
        business = {"source_name": within_source}
        filters = _within_source_filters(user, within_source)
    else:
        filters = build_filters(user, query, business=metadata_filtering)
    # 置信策略需要原始 Dense/BM25 信号，因此正常检索也收集有限候选快照；
    # 完整 trace 仍只在 explain=True 时对外返回，不进入模型上下文。
    stages: list[RetrievalStage] = []
    nodes = _retrieve_nodes(
        user, retrieval_query, mode=mode, num_queries=num_queries,
        fetch_k=fetch_k, filters=filters, business=business, stages=stages)

    # 业务 metadata 是增强，不是权限兜底。规则过滤零命中时只移除业务条件，
    # 权限过滤仍然保留，避免误抽取让一个本来有答案的问题静默变成无答案。
    if not nodes and business and not within_filename and not within_source:
        nodes = _retrieve_nodes(
            user, retrieval_query, mode=mode, num_queries=num_queries,
            fetch_k=fetch_k, filters=_visibility_filters(user), business={},
            stages=stages, stage_suffix="（移除业务 metadata 后回退）")

    def trace() -> RetrievalTrace:
        return RetrievalTrace(
            query=query, retrieval_query=retrieval_query, mode=mode,
            fetch_k=fetch_k, business_filters=business,
            reranker=reranker if rerank_pool > 0 else None,
            mmr_lambda=mmr_lambda if use_mmr else None, stages=stages,
        )

    def debug_payload():
        internal_trace = trace()
        signals = extract_retrieval_signals(internal_trace)
        return internal_trace if explain else None, signals

    if not nodes:
        retrieval_trace, signals = debug_payload()
        return SearchResult(evidence=[], note="没有命中任何内容。",
                            retrieval_trace=retrieval_trace,
                            retrieval_signals=signals)

    before = len(nodes)
    started = perf_counter()
    nodes = deduplicate_nodes(nodes)
    _add_stage(stages, "确定性去重", "沿用前序分数", nodes, started,
               f"按 node_id 和规范化正文去重，移除 {before - len(nodes)} 条")

    if rerank_pool > 0 and len(nodes) > 1:
        # cross-encoder 只负责重排，不在这里截到 top-k；完整候选池继续交给 MMR
        # 做集合级多样性选择。送入的仍是原始问题，不是 HyDE 假答案。
        started = perf_counter()
        nodes = RERANKERS[reranker](len(nodes)).postprocess_nodes(nodes, query_str=query)
        _add_stage(stages, "Cross-encoder 重排", "cross-encoder", nodes, started,
                   f"模型：{get_config().rerank_ce_model}")

    if use_mmr:
        started = perf_counter()
        _hydrate_embeddings(nodes)
        nodes = mmr_select(nodes, min(len(nodes), fetch_k), lambda_mult=mmr_lambda)
        _add_stage(stages, "MMR 多样性选择", "前序相关性分数", nodes, started,
                   "MMR 用相关性排名与候选间冗余决定顺序；显示分数不是 MMR 目标值")

    started = perf_counter()
    evidence = _evidence_from_ranked_nodes(
        user, nodes, top_k, expand_parent=expand_parent,
        # 消融证明收益来自 parent-aware 去重；无论是否展开正文都启用。
        deduplicate_parent=True)
    if stages is not None:
        stages.append(RetrievalStage(
            name="Parent 展开与最终截断", score_kind="最终证据分数",
            candidates=[RetrievalCandidate(
                rank=index, node_id=f"evidence:{index}", source_name=item.source_name,
                location=item.location, snippet=item.snippet[:500], score=item.score,
            ) for index, item in enumerate(evidence, start=1)],
            latency_ms=round((perf_counter() - started) * 1000, 2),
            note=f"parent-aware 去重后取 top-{top_k}",
        ))
    retrieval_trace, signals = debug_payload()
    if signals.decision == "no_evidence":
        return SearchResult(
            evidence=[],
            note=f"没有可回答该问题的可靠证据：{signals.decision_reason}",
            retrieval_trace=retrieval_trace,
            retrieval_signals=signals,
        )
    return SearchResult(
        evidence=evidence, retrieval_trace=retrieval_trace,
        retrieval_signals=signals)


def rebuild_lexical_index(batch_size: int = 500) -> int:
    """为已有 LlamaIndex 节点重建 sparse 索引；不重新切分、不调用 embedding。"""
    from ...db import vector_index_db

    with vector_index_db() as cur:
        cur.execute("SELECT node_id FROM data_nodes ORDER BY id")
        node_ids = [str(row["node_id"]) for row in cur.fetchall()]
    lexical.clear_all()
    for start in range(0, len(node_ids), batch_size):
        nodes = get_index().vector_store.get_nodes(
            node_ids=node_ids[start:start + batch_size])
        lexical.index_nodes(nodes)
    return len(node_ids)


def drop_all() -> None:
    """清空这一版的存储。评测脚本每轮开始前调，保证起点干净。"""
    global _index, _manifest_validated
    drop_vector_table(SCHEMA, TABLE)
    lexical.clear_all()
    with vector_index_db() as cur:
        cur.execute("TRUNCATE parent_contexts")
    clear_manifest()
    _index = None
    _manifest_validated = False
