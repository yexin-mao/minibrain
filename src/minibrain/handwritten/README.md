# handwritten/ —— 被框架换下来的手写实现

**这里的代码不在主路径上，但没有删。**

主路径现在是框架版：

| 环节 | 主路径（框架） | 这里（手写） |
|---|---|---|
| Agent 循环 | `agent/graph.py` — LangGraph | `agent_loop.py` |
| 切分 | `SentenceSplitter` | `chunking.py` |
| 向量检索 | `PGVectorStore` + `VectorIndexRetriever` | `vector_search.py` |
| 关键词 | `BM25Retriever` | `keyword.py` |
| 融合 | `QueryFusionRetriever`（RRF 模式） | `fusion.py` |
| 重排 | （待接 postprocessor） | `rerank.py` |

## 为什么保留

**一、评测基线。**

换框架到底换来了什么、代价是什么，只有两版跑同一批题才说得清。
两边的返回类型是同一个（`SearchResult` / `AnswerResult`），
所以 `scripts/` 下的评测脚本不改一行就能同时量两版。

**二、这些实现背后有实验数据。**

`eval/RESULTS.md` 里十三个探针，大半是围着这些文件转的：

- BM25 为什么不用 jieba：二元组会产生「偶然稀有词」（「三年年假」切出「年年」），
  污染 IDF —— 所以只把标识符类 token 放进倒排索引
- RRF 为什么不能改成加权求和：余弦是 0~1、BM25 无上界，量纲不同没有可比性
- 倒排索引比内存全量算快 **87.6 倍**
- pgvector 的 HNSW 索引会因反复 upload/purge 膨胀到 1125MB（表仅 18MB），
  优化器悄悄不用它、召回跟着掉，**两件事都不报错**
- 切分消融：800 字符最优，overlap 设 0 反而更好（和通行建议相反）

删掉这些文件，上面每一条都会变成「我记得好像是这样」。

**三、框架接不住的地方要有参照。**

最典型的一条：`BM25Retriever` 要求**全部节点在内存里**，
而 `keyword.py` + `vector_search.py` 专门为此建了 Postgres 倒排索引。
语料大了之后这个差别是数量级的。

## 和主路径共用的东西

搬迁时刻意留在主路径的三个文件，因为**两版都要用**：

- `modules/vector_rag/tokenizer.py` —— 中文怎么切。框架接管了 BM25 算法，
  但没接管「中文没有空格」这个问题，`BM25Retriever` 的分词器就是它
- `modules/vector_rag/embeddings.py` —— MRL 截断（4096→1024）+ 重新归一化。
  两版必须用**字节级相同**的向量，否则比出来的差异说明不了任何问题
- `agent/prompt.py` / `agent/types.py` —— 系统提示词是消融实验的产物
  （43 题 × 4 版本 × 3 轮 → 97.7%）；返回类型决定评测脚本能不能两版通用

## 怎么跑手写版

```python
from minibrain.handwritten import vector_search, agent_loop

vector_search.search(user, "问题", top_k=5, mode="hybrid")
agent_loop.answer(user, "问题")
```

两者的签名和主路径一致，可以直接对照跑。
