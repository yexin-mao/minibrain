# Embedding 模型对比

运行日期：2026-09-02。原始结果见 [`embedding-models.json`](embedding-models.json)。

## 结论

锁定同一批 130 篇文档、719 个 child chunks、58 道查询和 document-level
去重后，BGE-M3 在这套中文企业语料上优于当前 Qwen3-Embedding-8B 的
4096→1024 MRL 版本，并且本次 API 编码更快。

| Arm | 维度 | MRR | Recall@5 | Complete Recall@5 | NDCG@5 | 语料编码总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Embedding-8B MRL | 1024 | 0.846 | 0.743 | 0.569 | 0.726 | 128.38s |
| BGE-M3 native | 1024 | **0.919** | **0.796** | **0.586** | **0.791** | **39.52s** |

OpenAI text-embedding-3-small 作为第三个 arm 被当前地区策略拒绝（HTTP 403），
脚本将它记为失败而没有伪造或丢弃整场结果。

## 决策

BGE-M3 进入迁移候选，但暂不直接修改生产默认值。切换前还要补两项：

1. 在固定版本的 NanoNQ / NanoHotpotQA 上做 no-regression；
2. 重建独立影子索引，跑端到端答案质量和权限回归。

本实验是“当前生产输入口径下的 drop-in 对比”，没有给各模型添加专属 query
instruction；API 延迟也包含供应商排队和网络波动，不能视为稳定服务 SLO。
