"""GraphRAG 链路：LlamaIndex `PropertyGraphIndex`。

**动手之前先写了假设书**：`eval/GRAPHRAG_HYPOTHESIS.md`（git 提交时间早于本文件）。
那里写清了预期在哪赢、在哪不该赢、以及判据。看结论请先看它。

## 和向量链路的关系：受控对比，只换索引结构

刻意保持三样东西完全一致，否则测出来的差异说明不了任何问题：

| 变量 | 怎么锁的 |
|---|---|
| **语料** | 同一个 `eval/corpus/` |
| **embedding** | 复用 `vector_rag.chain.MinibrainEmbedding`（含 MRL 截断 4096→1024） |
| **评测** | 同一批探针、同一套 `ir_metrics` |

★ 这和当初把项目自己的 embedding 包成 `BaseEmbedding` 是同一个道理：
  **换框架/换索引时必须锁住不参与比较的变量。**

## 为什么不用 LightRAG（对标项目用的是它）

`companybrain/modules/graph-rag` 用 `lightrag-hku` + Neo4j。这里不跟，
因为换存储会引入额外变量——GraphRAG 用 Neo4j + 它自己的 embedding，
而向量链路用 pgvector + MRL 截断，那差异里就混进了「存储不同、向量不同」。

## 图是怎么建出来的

`SimpleLLMPathExtractor` 对**每个切片**调一次 LLM，让它抽出三元组：

    (张敏, 隶属, 后端组)
    (后端组, 隶属, 技术部)
    (技术部, 负责人, 李伟)

★★ **这就是 GraphRAG 最大的成本**，而且是数量级的差别：
   向量入库调 **0 次** LLM（只调 embedding），
   建图要调 **切片数** 次。130 篇语料切出约 900 片 → 约 900 次 LLM 调用。
   这个数字必须报进结论，见假设书里「成本必须一起量」那节。

`ImplicitPathExtractor` 是白送的：它不调 LLM，只把「同一篇文档的相邻切片」
连起来（`PREVIOUS`/`NEXT` 关系），用来保留原文顺序。

## 检索怎么走

`VectorContextRetriever`：先用向量找到最相关的**实体节点**，
再沿着图**向外走 `path_depth` 跳**，把路径上的关系和原文一起带回来。

所以它和纯向量检索的区别是：**向量检索找"最像的片段"，
图检索找"最像的实体，以及和它相连的东西"**。多跳问题的收益应该来自后半句。

## ⚠ 已知不干净的地方：权限

图谱天然跨文档——一个实体节点可能由五篇文档共同构成，而那五篇可能属于
不同的人、有不同的可见性。「这条关系用户 A 能不能看」在图上没有向量库
那么好回答（向量库是一个片段对应一个 owner）。

**本模块不解决这个问题**，评测在单用户下跑。这是 GraphRAG 上生产的真实
拦路虎，写在这里而不是假装不存在。假设书里也记了同一条。
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

from llama_index.core import PropertyGraphIndex, Settings, StorageContext
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor, SimpleLLMPathExtractor, VectorContextRetriever,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import Document
from llama_index.vector_stores.postgres import PGVectorStore

from ...config import get_config
from ...contracts import Evidence, ModuleError, SearchResult, UserContext
from ..vector_rag.chain import MinibrainEmbedding, _ensure_llm, _pg_params

MODULE_ID = "graph-rag"

# 图存到磁盘。★ 刻意不用 Neo4j：见 docstring「为什么不用 LightRAG」。
#   SimplePropertyGraphStore 是内存图 + JSON 持久化，够跑评测，
#   但它不支持并发写、也没有权限模型——**不是生产方案，是实验装置**。
STORE_DIR = pathlib.Path(__file__).resolve().parents[4] / "eval" / "results" / "graph_store"

_index: PropertyGraphIndex | None = None

# 每个切片最多抽几条三元组。调大→图更密、更贵；调小→漏关系。
# 10 是 LlamaIndex 的默认值，本次不调——**调参需要独立验证集**，
# 拿评测集调出来的参数是自欺（这条规矩全项目一致）。
MAX_PATHS_PER_CHUNK = 10


# ★★★ 中文抽取提示词。**必须自己写，框架默认那个会毁掉整个图。**
#
# LlamaIndex 的 DEFAULT_KG_TRIPLET_EXTRACT_PROMPT 是英文的，例子也是英文的
# （"Alice is Bob's mother" → "(Alice, is mother of, Bob)"）。
# 拿它去处理中文语料，实测模型会**把实体翻译成英文**：
#
#     输入：组长：张敏。共 8 人。隶属技术部。
#     输出：(Backend group, has leader, Zhang Min)
#           (Backend group, belongs to, Technology Department)
#
# 而且**解析函数认得出这个格式，所以不报错**——图正常建起来，
# 节点叫 `Backend group`、`Zhang min`。但用户提问用的是「后端组」「张敏」，
# 检索时对不上，**图等于白建，而且没有任何报错提示你**。
#
# 这是本项目第三次撞上同一类问题（前两次：BM25Retriever 默认按空白分词、
# QueryFusionRetriever 的改写提示词是英文的）：
#
# > **框架的默认值是给英文语料调的。中文场景下，凡是涉及"让模型输出什么"
# > 的默认提示词，都要假定它需要重写。**
_TRIPLET_PROMPT_ZH = (
    "下面是一段中文文本。请从中抽取最多 {max_knowledge_triplets} 条知识三元组，"
    "形式为 (主体, 关系, 客体)。\n\n"
    "硬性要求：\n"
    "- **主体、关系、客体全部用中文原文里的词，绝对不要翻译成英文**\n"
    "- 保留原文中的人名、部门名、编号、产品型号，不要改写不要意译\n"
    "- 每行一条，不要编号，不要解释\n\n"
    "示例：\n"
    "文本：张敏是后端组的组长，后端组隶属技术部。\n"
    "三元组：\n"
    "(张敏, 是组长, 后端组)\n"
    "(后端组, 隶属, 技术部)\n\n"
    "文本：{text}\n"
    "三元组：\n"
)


VECTOR_SCHEMA = "mod_graph_li"


def _entity_vector_store() -> PGVectorStore:
    """实体向量的存放处。**必须显式指定，否则重启后图就废了。**

    ★★ 这是踩出来的，而且很隐蔽：

      `SimplePropertyGraphStore.supports_vector_queries` 是 **False**，
      所以 PropertyGraphIndex 会另开一个 `SimpleVectorStore` 存实体向量。
      而 `graph_store.persist()` **只存图，不存那个向量库**。

      结果是：重启后图能加载回来（10 个实体、11 条三元组都在），
      但每个实体的 embedding 都是 None，`VectorContextRetriever`
      找不到任何入口节点，**检索恒定返回空，且不报错**。

      「存了但读不回来」是持久化最典型的半成品状态——
      本文件里已经栽了两次（上一次是 `_require_index` 忘了写加载路径）。

    放 PostgreSQL 而不是文件，理由和向量链路一致：不引入第二个存储系统。
    """
    import psycopg
    with psycopg.connect(get_config().database_url, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {VECTOR_SCHEMA}")
    return PGVectorStore.from_params(
        **_pg_params(), schema_name=VECTOR_SCHEMA, table_name="entities",
        embed_dim=get_config().embedding_dimensions,
        # search_path 要带 extensions，否则 type "vector" does not exist
        create_engine_kwargs={"connect_args": {
            "options": f"-c search_path={VECTOR_SCHEMA},extensions,public"}},
    )


def _storage() -> StorageContext:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    graph_store = SimplePropertyGraphStore()
    persist = STORE_DIR / "property_graph_store.json"
    if persist.is_file():
        graph_store = SimplePropertyGraphStore.from_persist_path(str(persist))
    return StorageContext.from_defaults(property_graph_store=graph_store,
                                        vector_store=_entity_vector_store())


def build(texts: dict[str, str], *, show_progress: bool = True) -> dict[str, Any]:
    """把语料建成图。返回成本统计——**这是本实验的核心产出之一**。

    参数是 {文件名: 正文}，不收 UserContext：本模块目前只服务评测，
    权限问题未解决（见 docstring）。

    ★ 返回的 `llm_calls_estimate` 是**估算**，不是精确计数：
      LlamaIndex 没有暴露调用计数器，这里用「切片数」近似
      （SimpleLLMPathExtractor 对每个切片调一次）。
      标成估算而不是假装精确——数量级对了就够支撑结论。
    """
    global _index
    _ensure_llm()

    cfg = get_config()
    splitter = SentenceSplitter(chunk_size=cfg.chunk_size,
                                chunk_overlap=cfg.chunk_overlap,
                                tokenizer=list)   # 字符口径，和向量链路一致
    docs = [Document(text=text, metadata={"filename": name})
            for name, text in texts.items()]
    nodes = splitter.get_nodes_from_documents(docs)

    started = time.perf_counter()
    _index = PropertyGraphIndex(
        nodes=nodes,
        embed_model=MinibrainEmbedding(),     # ★ 锁住 embedding 这个变量
        kg_extractors=[
            # 调 LLM 抽三元组 —— 成本全在这里
            SimpleLLMPathExtractor(llm=Settings.llm,
                                   # ★ 中文提示词，不用框架默认的，见 _TRIPLET_PROMPT_ZH
                                   extract_prompt=PromptTemplate(_TRIPLET_PROMPT_ZH),
                                   max_paths_per_chunk=MAX_PATHS_PER_CHUNK,
                                   num_workers=4),
            # 不调 LLM：把同一篇文档的相邻切片连起来，保留原文顺序
            ImplicitPathExtractor(),
        ],
        storage_context=_storage(),
        # ★ 实体向量存 PG，见 _entity_vector_store 的说明
        vector_store=_entity_vector_store(),
        show_progress=show_progress,
    )
    elapsed = time.perf_counter() - started
    _index.property_graph_store.persist(str(STORE_DIR / "property_graph_store.json"))

    graph = _index.property_graph_store
    # ★ 实体和三元组分开数：只看实体数会被"切片节点"混淆——
    #   PropertyGraphIndex 会把原文切片也作为节点存进图里。
    entities = [n for n in (graph.get() or [])
                if type(n).__name__ == "EntityNode"]
    triplets = graph.get_triplets(entity_names=[e.name for e in entities]) or []
    return {
        "documents": len(texts),
        "chunks": len(nodes),
        "llm_calls_estimate": len(nodes),     # 每切片一次，见 docstring
        "build_seconds": elapsed,
        "entities": len(entities),
        "triplets": len(triplets),
    }


def _require_index() -> PropertyGraphIndex:
    """拿到图索引。内存里没有就**从磁盘加载**，不重建。

    ★★ 第一版这里只检查内存里的 `_index`，新进程一律报「图还没建」。
      而建图要调 LLM **切片数** 次（130 篇语料约 900 次），
      **进程重启就重建是不可接受的**——这跟向量库重启后还在是同一个预期。

      持久化当时就做了（`build()` 末尾 persist），但**没有写加载路径**。
      「存了但读不回来」是持久化最典型的半成品状态：
      看起来有文件，实际等于没存。
    """
    global _index
    if _index is not None:
        return _index

    persist = STORE_DIR / "property_graph_store.json"
    if not persist.is_file():
        raise ModuleError("图还没建，先跑 build()", code="graph_not_built", status=503)

    _ensure_llm()
    _index = PropertyGraphIndex.from_existing(
        property_graph_store=SimplePropertyGraphStore.from_persist_path(str(persist)),
        vector_store=_entity_vector_store(),   # ★ 少了这个，实体向量读不回来
        embed_model=MinibrainEmbedding(),
    )
    return _index


def search(user: UserContext, query: str, top_k: int = 5,
           *, path_depth: int = 2) -> SearchResult:
    """和向量链路同签名同返回，同一套评测脚本能直接跑。

    ★ `user` 参数收下但**目前不用于过滤**——权限在图上没解决，
      见模块 docstring。签名保持一致是为了让评测脚本通用，
      不是暗示权限已经做好了。

    `path_depth=2`：从命中的实体向外走两跳。
    这个值直接决定「多跳能力」——1 跳退化成普通检索，跳得太远会引入噪声。
    """
    query = query.strip()
    if not query:
        raise ModuleError("查询不能为空", code="empty_query")

    index = _require_index()
    retriever = VectorContextRetriever(
        index.property_graph_store,
        vector_store=index.vector_store,
        embed_model=MinibrainEmbedding(),
        similarity_top_k=top_k,
        path_depth=path_depth,
        include_text=True,        # 把命中实体所在的原文一起带回来
    )
    nodes = retriever.retrieve(query)

    if not nodes:
        return SearchResult(evidence=[], note="图检索没有命中任何内容。")

    return SearchResult(evidence=[
        Evidence(
            module=MODULE_ID,
            source_name="graph",
            location=n.node.metadata.get("filename", "?"),
            snippet=n.node.get_content(),
            score=round(float(n.score), 4) if n.score is not None else None,
        )
        for n in nodes[:top_k]
    ])


def drop_all() -> None:
    """删掉持久化的图**和实体向量**。评测每轮开始前调，保证起点干净。

    ★ 两个都要删。只删图不删向量表，下次建图时旧实体向量还在，
      检索会命中已经不存在的实体——那种脏状态最难查。
    """
    global _index
    import psycopg
    persist = STORE_DIR / "property_graph_store.json"
    if persist.is_file():
        persist.unlink()
    with psycopg.connect(get_config().database_url, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {VECTOR_SCHEMA}.data_entities CASCADE")
    _index = None
