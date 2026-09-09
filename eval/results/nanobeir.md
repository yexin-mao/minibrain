# NanoNQ / NanoHotpotQA 公开检索评测

> 运行日期：2026-08-16。这是选定的两个 NanoBEIR 任务，不是完整 NanoBEIR 总分。
> 逐题排名和全部指标见 [`nanobeir.json`](nanobeir.json)。
> 这份结果是“候选池内 BM25”的历史基线；根据它修正 MMR 默认值后，
> 主路径已改为持久化全库 BM25。为避免在同一公开测试集上边改边调，尚未重跑；
> 下一次重跑应视为冻结新实现后的一次验证。

## 实验口径

- `NanoNQRetrieval` commit `e3973b405feb9e94d07fd971d690d24d7f45264a`：5035 documents / 50 queries。
- `NanoHotpotQARetrieval` commit `a38a2615f29adba8b47a42905031aa9c137d8fae`：5090 documents / 50 queries，每题 2 篇相关文档。
- embedding：`qwen/qwen3-embedding-8b`，4096 维截断、重新归一为 1024 维。
- 切分：800 字符 / 120 overlap；NanoNQ 产生 8695 chunks，NanoHotpotQA 产生 6573 chunks。
- 所有 arm 共用 40 条候选池，最终返回 top-10；业务 metadata filtering 关闭。
- chunk 排名在评分前聚合回原始 corpus document，每篇只保留最高排名 chunk。
- 查询 embedding 预热并在三个 arm 之间复用；延迟不含远程 query embedding。

## 结果

### NanoNQRetrieval

| arm | MRR | Recall@5 | Recall@10 | Complete Recall@5 | Complete Recall@10 | NDCG@5 | 中位延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | **0.619** | **0.790** | **0.870** | **0.760** | **0.860** | **0.643** | **94.3 ms** |
| hybrid | 0.591 | 0.770 | 0.830 | **0.760** | 0.820 | 0.627 | 236.6 ms |
| hybrid + MMR | 0.582 | 0.730 | 0.830 | 0.700 | 0.820 | 0.601 | 317.0 ms |

### NanoHotpotQARetrieval

| arm | MRR | Recall@5 | Recall@10 | Complete Recall@5 | Complete Recall@10 | NDCG@5 | 中位延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 0.925 | 0.840 | 0.870 | 0.720 | 0.740 | 0.835 | **72.8 ms** |
| hybrid | **0.926** | **0.900** | **0.930** | **0.820** | **0.860** | **0.866** | 184.5 ms |
| hybrid + MMR | 0.925 | 0.850 | **0.930** | 0.720 | **0.860** | 0.807 | 261.9 ms |

## 结论

1. **hybrid 不是全面优于 vector。** NanoNQ 上 MRR、Recall 和 NDCG 全部下降；
   NanoHotpotQA 上则把 Complete Recall@5 从 0.72 提高到 0.82，逐题是 5 胜 / 0 负 / 45 平。
   因此更准确的说法是：**当问题需要多文档证据时，当前 hybrid 有价值；对单证据 NQ 任务没有。**

2. **当前 MMR (`lambda=0.7`) 没有证明价值。** 它确实提高了结果中不同文档数：
   NanoNQ 9.32 → 9.58，NanoHotpotQA 9.48 → 9.62；但代价是两套数据的 top-5 相关性都下降。
   相对 hybrid，NanoNQ Complete Recall@5 是 0 胜 / 3 负，HotpotQA 是 0 胜 / 5 负。
   **多样性增加不等于证据更完整。**

3. **当前 MMR 不应默认开启或写成已证实的提升。** 合理的下一步是先把它改为默认关闭；
   如果还想保留，再在独立开发集上扫 `lambda`，不能直接拿这 100 题调参后再在同一批题上宣称提升。

4. **延迟代价明显。** hybrid 约为 vector 的 2.5–2.6 倍，MMR 进一步增加延迟。
   另外，两套语料的远程 embedding 入库分别用时 35.4 和 34.1 分钟；这是当前 16 条/批、4 路并发的离线吞吐问题，不应混入在线查询延迟。

## 已观察到的实现限制

- 框架版 BM25 只在向量候选池内工作，不是能从全库独立救回文档的 sparse retriever。
- 每个 hybrid 查询都重建 `BM25Retriever`，并产生 tokenizer 弃用警告；这一部分同时损害延迟和日志可读性。
- `SentenceSplitter` 会在切分时考虑 metadata 长度，评测用的可逆文件名和权限 metadata 会影响实际 chunk 数。
  本次是实际产品链路的结果，但下一轮切分对照应将这个变量单独拆出。
