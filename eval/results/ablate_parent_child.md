# 父子检索消融

同一份 800 字 child 索引、同一批 child 排名；三组依次拆分 parent-aware 去重与
1600 字 parent 正文展开。三组最终都经过生产的 4000-token ContextAssembler，
没有调用生成式 LLM。

| 指标 | flat child | child + parent 去重 | small-to-big |
|---|---:|---:|---:|
| Recall@5 | 0.847 | 0.886 | 0.882 |
| NDCG@5 | 0.869 | 0.894 | 0.889 |
| MRR | 0.955 | 0.955 | 0.955 |
| 严格 Complete Recall@5 | 0.812 | 0.812 | 0.812 |
| 关键答案字符串全部在 context | 0.556 | 0.556 | 0.556 |
| context token 中位数 | 2068 | 2068 | 2536 |
| context token P95 | 3943 | 3919 | 3721 |
| 触发 token budget 的题数 | 18 | 15 | 23 |
| 独立文档数中位数 | 11.0 | 11.0 | 11.0 |
| 投影延迟中位数 | 0.07 ms | 0.04 ms | 1.35 ms |

## 直接结论

- parent 让 context token 中位数变化 +469。
- 关键答案支持率变化 +0.000。
- 排除 parent 去重后，扩大正文自身让 Recall@5 变化 -0.004，
  关键答案支持率变化 +0.000。
- 检索只执行一次，因此 IR 差异来自 parent 合并和 token 预算，不是两次召回波动。

生产默认选择第二组：返回 child 正文，但用 parent_id 去重。完整 parent 展开通过
`EXPAND_PARENT_CONTEXT=true` 保留为实验开关，默认关闭。

> “关键答案字符串存在”是确定性支持度代理，不等于完整答案质量；若它有明确收益，
> 再对失败/改善样本跑少量 LLM 端到端评测，而不是一上来对 58 题多调用模型。
