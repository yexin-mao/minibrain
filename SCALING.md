# 这套做法在真实规模下会怎样

**这份文件回答一个尖锐的问题：把文档清单塞进 system prompt，在 15 篇时有效，
那 1000 万篇文档的企业怎么办？**

答案是：**不行，而且我知道不行**。下面把边界、业界做法、以及"为什么现在不做"写清楚。
一个方案的价值不只在它解决了什么，还在于**你是否知道它在哪里失效**。

---

## 一、现在的做法

路由准确率从 83.7% 提到 95.3%（见 [`eval/ROUTING.md`](eval/ROUTING.md)），
靠的是往 system prompt 里加两段东西：

```
可检索的文档（只列出你有权访问的）：
  dept-tech.md —— 技术部
  team-backend.md —— 后端组
  ...（15 篇全部列出）

权威来源规则（重要）：同一个事实可能在文档和表格里都出现，此时以下面的规定为准：
- 组织架构、岗位职责、制度规定、项目信息 → 以文档为准，用 vector_search
- 人数、金额、日期等需要统计计算的 → 以表格为准，用 table_query
- 表格里的行可能包含记账用的辅助行，不代表真实的组织单元。
```

## 二、它在什么规模下崩

| 语料规模 | 文档清单占的字符 | 结论 |
|---|---|---|
| 15 篇（现在） | ~300 | 可行 |
| 1,000 篇 | ~2 万 | 勉强，但每次提问都要重发 |
| 100,000 篇 | ~200 万 | **超出任何模型的上下文窗口** |
| 10,000,000 篇 | ~2 亿 | 荒谬 |

**这一段是 `O(文档数)` 的。它不可扩展，没有任何辩解余地。**

### 但消融数据已经把两段的贡献拆开了

回头看 [`eval/PROMPT_ABLATION.md`](eval/PROMPT_ABLATION.md) 那张表：

| 加了什么 | 目标类别正确率 | 随文档数怎么增长 |
|---|---|---|
| 文档清单 | 39% → 78% | **O(文档数)** ← 不可扩展 |
| 权威来源规则 | 78% → **100%** | **O(1)**，永远就那几行 ← 可扩展 |

**贡献更大的那一段，恰好是可扩展的那一段。**

所以真正要解决的问题被缩小了：**只需要给"文档清单"（贡献 39pp）找一个可扩展的替代品。**

---

## 三、业界怎么做

### ① 语义路由（Semantic Router）—— 根本不进 prompt

预先给每条链路准备一批典型问题，转成向量存起来。用户提问时把问题也转成向量，
**跟这些样例做相似度匹配**，最近的那条就是要走的链路。

```
「后端组组长是谁」→ 向量化 → 与各链路样例比对 → 最接近"文档类"样例 → 走文档链路
```

- **完全不占 prompt 空间**，与文档数无关
- 比让模型读提示词做判断快得多——有实测报告称约 **50 倍**的时间优势，准确率也更高
- 提示词路由用久了会退化：明明在 prompt 里给定了类别名，模型仍会自己改名字

### ② 分层索引（Hierarchical RAG）—— 路由到"领域"，不是"文档"

不列 1000 万篇文档，而是列几十个领域：

```
可检索的知识域：
  人事制度（12,400 篇）
  财务报销（3,200 篇）
  项目文档（180,000 篇）
```

检索自上而下：**先定领域 → 再定文档集合 → 最后到文档**。
prompt 里的东西变成 `O(领域数)`。高层节点代表宽泛概念，低层节点存细节内容，
早期就把不相关的分支剪掉，噪声大幅下降。

### ③ 元数据过滤 —— 先剪枝再算相似度

给文档打标签（部门、日期、文档类型），检索前先按标签砍掉范围。
多级过滤能大幅减少每次查询需要比较的向量数量。

### ④ 语义层（Semantic Layer / Semantic Model）

**★ 我手写的那段"权威来源规则"，在业界有正式名字。**

Snowflake 的 Cortex Analyst 用一份 **YAML 语义模型**告诉系统
"哪些字段是什么含义、该怎么用"，然后才翻译成 SQL。

也就是说，**我是重新发明了一个已有的东西**。
但区别在于：我是先测出 `doc-08` 修不好、再做消融实验、才知道这东西非有不可——
而不是因为文档里写了所以照抄。

---

## 四、和产业形态的对照

**minibrain 的结构和 Snowflake Cortex Agent 是同构的：**

| | minibrain | Snowflake |
|---|---|---|
| 非结构化链路 | `vector-rag` | Cortex Search |
| 结构化链路 | `table-rag`（受限只读 SQL） | Cortex Analyst（text-to-SQL） |
| 路由层 | 手写 tool loop | Cortex Agent |
| 消解字段歧义 | system prompt 里的权威来源规则 | YAML 语义模型 |

**Databricks 的 Genie 则只有结构化那一条**——不能回答 PDF、Word 这类文件的问题。
所以它比 Snowflake 少一条链路。

一个值得知道的限制：这两家的产品**只读自己平台的元数据、只在自己平台的算力上执行**，
跨平台整合仍是未解决的问题。

---

## 五、第二个可扩展性问题：模型不知道列里有哪些取值

`tbl-18` 那个回归（见 [`eval/ROUTING.md`](eval/ROUTING.md)）：模型写
`WHERE "状态" = '驳回'`，而实际值是 `'已驳回'`，返回 0。

根因在 `table_rag.describe_schema`：

```python
cols = ", ".join(f'"{c["name"]}" {c["type"]}' for c in t["columns"])
#                        ↑ 名字        ↑ 类型
#                   只有这两样，从头到尾没碰过表里的数据
```

而 `columns` 是**入库那一刻**从 CSV 表头和类型推断写死的，
**从来没有 `SELECT DISTINCT` 看过实际取值**。模型只能猜枚举值。

### 这在学术上有名字

text-to-SQL 领域称之为 **schema linking**（模式链接），
其中专门处理取值的部分叫 **value linking / value retrieval**。

这是该领域公认的瓶颈：SQL 生成之前，系统必须先确定一份**足够小但足够全**的 schema 上下文。
大规模数据库上，把完整 schema 塞进去既会超出上下文窗口，又会引入噪声干扰模型。
2025~2026 年有多篇论文专门做这件事（LinkAlign 针对上千字段的多库场景、
SchemaGraphSQL 用图算法做路径查找、EviLink 用不确定性引导的证据获取）。

**所以我踩到的不是一个小 bug，是这个领域核心难题的最小版本。**

### 分规模的修法

| 规模 | 做法 |
|---|---|
| 小表（现在） | 对低基数文本列直接 `SELECT DISTINCT`，全部注入 |
| 大表 | 只注入 top-N 高频值 + 基数统计 |
| 超大规模 | **按需检索**：用问题里的关键词去值索引里查，只注入匹配上的（value retrieval） |

---

## 六、为什么现在不做这些

不是不知道，是**做了也测不出来**。

| 方案 | 触发条件 | 现在够吗 |
|---|---|---|
| 语义路由 | 需要证明"prompt 里的清单撑不住了" | ❌ 15 篇，清单占 300 字符 |
| 分层索引 | 需要多个领域、足够多的文档 | ❌ 语料只有一个领域 |
| 元数据过滤 | 需要文档量大到相似度计算成本可感知 | ❌ **实测余弦只占单次检索的 0.2%**（见 RESULTS.md 探针七） |
| **value linking** | ✅ `tbl-18` 已经实测失败 | ✅ **可以做** |

**语料规模是这些方案的前置条件。**
15 篇 1951 字符的语料上，向量检索的 `MRL = 1.000`、`Hit Rate@k = 1.000`——
指标已经顶到天花板，任何检索侧的优化都测不出差别。

**要做规模化方案，必须先扩语料。** 这是从数字里读出来的，不是拍脑袋。

---

## 七、这份文件回答的面试问题

| 问题 | 答案 |
|---|---|
| 你这个方案能扩展吗？ | 不能。清单那段是 `O(文档数)`。但消融数据显示贡献更大的规则是 `O(1)`，需要替换的只是清单 |
| 那该怎么做？ | 语义路由（不占 prompt、快约 50 倍）+ 分层索引（路由到领域而非文档）+ 元数据过滤 |
| 为什么不现在就做？ | 15 篇语料上 MRR 已经是 1.000，做了测不出差别。要做必须先扩语料 |
| 和业界方案比差在哪？ | 结构和 Snowflake Cortex Agent 同构；他们用 YAML 语义模型消歧，我手写在 prompt 里 |
| 遇到过什么难题？ | 模型猜错了枚举值。这在 text-to-SQL 里叫 value linking，是该领域公认瓶颈的最小版本 |

---

## 参考

- [LinkAlign: Scalable Schema Linking for Real-World Large-Scale Multi-Database Text-to-SQL](https://arxiv.org/abs/2503.18596v3)
- [SchemaGraphSQL: Efficient Schema Linking with Pathfinding Graph Algorithms](https://arxiv.org/pdf/2505.18363)
- [EviLink: Multi-Path Schema Linking with Uncertainty-Guided Evidence Acquisition](https://arxiv.org/pdf/2605.29670)
- [Semantic Routing for Enhanced Performance of LLM-Assisted Intent-Based Management](https://arxiv.org/pdf/2404.15869)
- [Mastering RAG Chatbots: Semantic Router — User Intents](https://medium.com/@talon8080/mastering-rag-chabots-semantic-router-user-intents-ef3dea01afbc)
- [Routing in RAG Driven Applications — Towards Data Science](https://towardsdatascience.com/routing-in-rag-driven-applications-a685460a7220/)
- [Hierarchical RAG Architecture for Large Document Collections — ML Journey](https://mljourney.com/hierarchical-rag-architecture-for-large-document-collections-scaling-information-retrieval-for-enterprise-applications/)
- [Scaling RAG to 20M Docs: Challenges & Solutions](https://www.chitika.com/scaling-rag-20-million-documents/)
- [Building AI Applications with Snowflake Cortex: RAG, Text-to-SQL & CoCo](https://www.snowflake.com/en/developers/guides/accelerate-app-dev-coco/)
- [Snowflake Cortex Analyst vs Databricks Genie](https://colrows.com/blogs/cortex-analyst-vs-genie/)


---

## 附：延迟实测推翻了本文档的一个假设

本文档多处提到"内存全量余弦在几万条以上会变慢"，隐含假设是**余弦计算是瓶颈**。

**实测不是**（`eval/RESULTS.md` 探针七）：

| | 占单次检索 | 占本地计算 |
|---|---|---|
| embedding 网络往返 | **98.2%** | — |
| SQL 拉取全部片段 | 1.6% | **90.1%** |
| BM25 | 0.2% | 9.6% |
| **余弦计算** | **0.0%** | **0.2%** |

所以 pgvector 的价值**不在"算得快"，在"不用把全部数据搬出来"**——
瓶颈是数据搬运，不是向量运算。

线性外推：本地计算要到约 **4,700 个片段**才追平网络延迟。

### 900 片段下复测（探针十）：瓶颈换人了

| | 84 片段 | 900 片段 | 增长 |
|---|---|---|---|
| SQL 拉取 | 9.41ms | 102.24ms | 10.9x |
| **BM25** | 4.98ms | **201.65ms（5条）** | **40.5x 超线性** |
| **余弦** | 0.10ms | 0.42ms（5条） | 4.2x |

**余弦占本地计算 0.1%，BM25 占 28.2%，SQL 拉取占 71.6%。**

BM25 超线性的根因：`bm25_scores` **每次查询都把全部文档重新分词、重算 IDF**——
是 O(总字符数) 不是 O(片段数)。生产系统用预建倒排索引。

**优化顺序因此完全反了过来：**

1. BM25 预建倒排索引（增长最快，根因明确）
2. SQL 只取需要的列/行（占本地 71.6%）
3. embedding 查询缓存（占端到端 82.4%）
4. pgvector + HNSW（**余弦只占 0.1%，收益最小**）

pgvector 真正的价值是第 2 项——让数据库只返回 top-k，
**不用把 900×1024 个浮点数搬进内存**。是"减少数据搬运"不是"加速向量运算"。
