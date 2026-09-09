# Cross-encoder rerank 消融

> 运行日期：2026-08-17。模型：`BAAI/bge-reranker-base`。
> 这是本地专用 cross-encoder，不是已经删除的 LLM listwise rerank。

## 结论

**CE 有稳定的排序收益，但当前不应默认开启。**

- 中文企业集 40 候选上，Complete Recall@5、NDCG@5、MRR 都提升；
- 两个公开集的冻结 top-10 候选内复核方向一致；
- 但企业集检索 P50 从 14ms 增至 1467ms（约 107 倍），CE 选中的前五片段
  中位字符数从 1062 增至 2655，甚至高于 baseline 前十的 1700；
- 它证明了“能排得更准”，没有证明“能用更少 context 达到同样效果”。

因此生产默认继续使用 hybrid + 确定性去重，CE 保留为检索实验室开关。

## 一、中文企业语料：完整 40 候选实验

- 130 篇文档，58 题；业务 metadata 和 MMR 均关闭；
- baseline 与 CE 使用同一 40 条候选池；
- 每组只运行一次 top-10，@3/@5 是同一排名的前缀；
- 查询 embedding 预热，延迟不含远程 embedding；
- CE 模型冷启动不计入逐题延迟，进程内权重只加载一次。

| 配置 | Complete Recall@3 | Complete Recall@5 | Recall@5 | NDCG@5 | MRR | context@5 | context@10 | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.672 | 0.707 | 0.869 | 0.886 | 0.955 | 1,062 | 1,700 | 14ms | 26ms |
| CE | **0.707** | **0.741** | **0.873** | **0.912** | **0.971** | 2,655 | 5,264 | 1,467ms | 1,571ms |

Recall 胜/负/平：@3 为 `6/4/48`，@5 为 `9/4/45`，@10 为 `9/1/48`。

逐题看，CE 更擅长产品型号、工单编号和术语定义；它会伤害一部分组织关系与全局聚合题，
例如“公司一共有几个部门”和“陈刚和刘洋是同一个部门吗”。因此不能根据平均提升直接
推导出“所有查询默认开启”。

### 两把事先固定的尺子

1. **等片数**：CE@5 的 Complete Recall 0.741 高于 baseline@5 的 0.707，胜出。
2. **等实际成本**：CE@5 追平 baseline@10 的 Complete Recall 0.741，但 context
   是 2655 vs 1700 字，且多约 1.45 秒，失败。

## 二、公开集：冻结 top-10 候选内复核

为了避免重复计算约一万篇 corpus embedding，公开集使用已经冻结的
`nanobeir-global-bm25.json` hybrid top-10 文档候选。候选原文重新经过当前 child
chunking，CE 对 chunk 打分后聚合回文档排名。

| 数据集 | 指标 | baseline | CE | Recall@5 胜/负/平 | CE P50 / P95 |
|---|---|---:|---:|---:|---:|
| NanoNQ | Complete Recall@5 | 0.720 | **0.760** | 6 / 1 / 43 | 465 / 660ms |
|  | Recall@5 | 0.730 | **0.810** |  |  |
|  | NDCG@5 | 0.585 | **0.695** |  |  |
|  | MRR | 0.557 | **0.683** |  |  |
| NanoHotpotQA | Complete Recall@5 | 0.800 | **0.820** | 2 / 0 / 48 | 374 / 470ms |
|  | Recall@5 | 0.890 | **0.910** |  |  |
|  | NDCG@5 | 0.859 | **0.921** |  |  |
|  | MRR | 0.926 | **1.000** |  |  |

### 限制

公开集这一组是**候选内排序验证，不是完整端到端 40 候选重跑**。CE 无法引入冻结
top-10 之外的文档，因此不能用它声称 11–40 名的正例被救回。完整原始排名与限制字段
保存在 `nanobeir-ce-saved-top10.json`。

## 三、实验同时发现并修复的问题

1. 原实现每个查询重新构造 CE；第一次缓存又误按 `top_n`，候选数 39/40 仍会加载多份
   1.1GB 权重。现在使用单一底层模型，浅拷贝 postprocessor 只改变本次 `top_n`。
2. 企业消融脚本仍 patch 迁移前的 embedding 引用，实际每个 arm 都在远程调用 query
   embedding。现在 patch LlamaIndex 真正使用的 `llamaindex_setup.embed_query`。
3. 原矩阵把 CE 和 MMR 同时打开，无法归因。现在主对照只改变 CE 一个变量。
4. 原矩阵为 @5/@10 重复执行同一 CE 排名。现在一次 top-10，两个 cutoff 共用排名。

原始结果：`ablate_rerank_ce.json`、`nanobeir-ce-saved-top10.json`。
