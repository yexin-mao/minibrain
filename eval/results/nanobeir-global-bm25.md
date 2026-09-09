# NanoBEIR 冻结版本验证：全局 BM25

> 运行日期：2026-08-17。这是选定的 NanoNQ / NanoHotpotQA 两个任务，不是完整 NanoBEIR 总分。
> 全部指标和逐题排名见 [`nanobeir-global-bm25.json`](nanobeir-global-bm25.json)；
> 历史候选池内 BM25 基线见 [`nanobeir.json`](nanobeir.json)。

## 实验口径

- `NanoNQRetrieval`：5035 documents / 50 queries；当前切分产生 6371 chunks。
- `NanoHotpotQARetrieval`：5090 documents / 50 queries；当前切分产生 5375 chunks。
- 每个 arm 使用相同的 40-document 候选规模，最终返回 top-10，关闭业务 metadata filtering。
- hybrid 由全库 dense 与全库 sparse BM25 独立召回，再用 RRF 融合。
- chunk 排名在评分前聚合回原始 document；查询 embedding 不计入延迟。
- 当前结果相对历史基线同时包含两项改动：全局 BM25，以及“先切正文、后附 metadata”的分块修复，不能把全部差异只归因于 BM25。

## 当前冻结版本结果

### NanoNQRetrieval

| arm | MRR | Recall@5 | Recall@10 | Complete Recall@5 | Complete Recall@10 | NDCG@5 | 中位延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | **0.571** | **0.770** | **0.890** | **0.740** | **0.880** | **0.605** | **63.8 ms** |
| global hybrid | 0.557 | 0.730 | 0.840 | 0.720 | 0.800 | 0.585 | 92.5 ms |
| global hybrid + MMR | 0.530 | 0.730 | 0.810 | 0.700 | 0.780 | 0.565 | 173.2 ms |

### NanoHotpotQARetrieval

| arm | MRR | Recall@5 | Recall@10 | Complete Recall@5 | Complete Recall@10 | NDCG@5 | 中位延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 0.923 | 0.840 | 0.870 | 0.720 | 0.760 | 0.836 | **51.4 ms** |
| global hybrid | **0.926** | **0.890** | **0.910** | **0.800** | **0.820** | **0.859** | 78.0 ms |
| global hybrid + MMR | 0.923 | 0.840 | **0.910** | 0.700 | **0.820** | 0.809 | 162.0 ms |

## 与历史实现的对比

- NanoNQ：当前 vector 的 MRR 从 0.619 降至 0.571，hybrid 从 0.591 降至 0.557；当前 global hybrid 仍未超过 vector。
- NanoHotpotQA：global hybrid 相对 vector 仍明显改善多证据完整度，Complete Recall@5 为 0.80 vs 0.72；但略低于历史 hybrid 的 0.82。
- 在线延迟明显改善：hybrid 中位延迟由 236.6 / 184.5 ms 降至 92.5 / 78.0 ms，分别降低约 61% / 58%。
- 分块修复使 chunks 从 8695 / 6573 降至 6371 / 5375，离线入库时间从 35.4 / 34.1 分钟降至 32.2 / 23.9 分钟。

## 结论

1. **默认仍应使用 vector。** 在通用单证据 NanoNQ 上，global hybrid 没有证明质量收益，还增加约 45% 查询延迟。
2. **hybrid 适合作为多证据查询策略。** NanoHotpotQA 上，它把 Complete Recall@5 从 0.72 提高到 0.80，说明 sparse + dense 对多跳证据覆盖仍有价值。
3. **MMR 保留为可选能力但默认关闭。** 两个任务都没有稳定增益，而且大约再增加 80–84 ms。
4. **全局 sparse 索引的主要确定收益是架构正确性和速度，不是公开集上的全面质量提升。** 它能从全库独立召回、无需每次查询重建 BM25，且本次延迟下降超过一半。
5. **不再用这 100 题继续调参。** 若要决定自动路由或调 RRF/MMR，应另设开发集，再保留这次结果作为冻结验证记录。

评测结束后的清理检查：`data_nodes / node_lexical_stats = 76 / 76`，sparse 孤儿记录 0，临时 benchmark 用户 0。
