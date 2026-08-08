"""重排（rerank）：把召回回来的候选重新排一次序。

## 为什么需要它——理由是从指标里读出来的，不是因为"大家都这么做"

58 道题实测（eval/RESULTS.md）：

    k=3    Recall 0.797   Complete Recall 0.655
    k=10   Recall 0.975   Complete Recall 0.948

`Complete Recall@k` = 一道题的**全部**必需文档都进了前 k，才算 1，差一篇就是 0。

**信息在 top-10 里，但没挤进 top-3。** 这不是召回问题（Recall@10 已经 0.975），
是**集中度**问题——而这正是重排干的活。

★ 但真实空间比表面小。58 题里有 12 题需要 ≥4 篇文档（最多一题要 11 篇），
  这些题在 k=3 时**数学上不可能**拿满 Complete Recall。算上天花板：

      k=3    上限 0.793   实测 0.655   真实空间 0.138
      k=5    上限 0.931   实测 0.724   真实空间 0.207   ← 空间最大
      k=10   上限 0.983   实测 0.948   真实空间 0.035

  **所以重排要盯的是 k=5，不是 k=3。** 不先算天花板就会把 0.655→0.948
  当成 29 个点的空间，那是虚的。

## 双塔 vs 交叉编码

- **双塔（bi-encoder）**：问题和文档**各自**编码成向量，再算余弦。
  快——文档向量可以预先算好存库里。但两边从没"见过面"，细粒度判断不了。
  这是当前 `_vector_search_in_db` 干的事。
- **交叉编码（cross-encoder）**：问题和文档**拼在一起**送进模型，直接出相关性分数。
  准得多，但每对都要过一次模型，**不可能拿它扫全库**。

所以标准做法是两段式：双塔召回 top-N（保召回）→ 交叉编码重排 top-k（修排序）。

本模块用 **LLM 做重排**，不引入本地 cross-encoder 模型。理由：
零新增依赖、复用已有的 `AGENT_*` 配置，先把链路和评测跑通。
如果评测证明重排有收益，再考虑换成专用 cross-encoder（更快更便宜）；
如果证明没收益，就省下了引入一个模型依赖的成本。

## listwise，不是 pointwise

- **pointwise**：每个候选单独打分 → N 次 API 调用，慢且贵
- **listwise**：所有候选一次性给模型，让它输出排序 → **1 次调用**

选 listwise。代价是受上下文长度限制，以及下面这个坑：

★★ **位置偏见（position bias）**：LLM 天然偏向排在前面的选项。
   如果直接按召回顺序喂进去，模型很可能"确认"原有排序 ——
   那样测出来的收益是假的，因为重排根本没做事。
   所以本模块**打乱候选顺序再送给模型**，拿到结果后映射回原候选。
   `probe` 里会对比"打乱 vs 不打乱"，把这个偏见的大小量出来。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from openai import OpenAI

from ..config import get_config
from ..contracts import ModuleError

_client: OpenAI | None = None

# 每个候选片段送进重排提示词时截断到多少字符。
# 太长会挤爆上下文也更贵；太短会丢掉判断依据。
# 800 字对应本项目的切分粒度（一个片段通常 300~600 字），基本不截。
SNIPPET_CHARS = 800

_PROMPT = """你在做检索结果重排。下面是用户的问题和若干候选片段。

请判断每个片段对**回答这个问题**有多大帮助，然后把片段编号按帮助从大到小排序。

判断标准：
- 片段直接包含答案 → 最高
- 片段包含回答所必需的中间事实（比如问"张敏的上级的上级"，
  而片段告诉你"张敏在后端组"）→ 同样重要，不要因为它没有直接答案就压低
- 片段只是词面相似但答不了这个问题 → 最低

只输出一个 JSON 数组，元素是片段编号，从最有帮助到最没帮助。
不要输出任何其他文字。例如：[3, 1, 5, 2, 4]

问题：{query}

候选片段：
{candidates}"""


@dataclass
class RerankOutcome:
    """重排结果。`fell_back` 是给评测用的——回落了就说明这次重排没真的生效。"""

    order: list[int]              # 原候选列表的下标，按重排后的顺序
    fell_back: bool               # True = 模型输出没法用，回落到了原顺序
    reason: str | None = None     # 回落原因，写进评测报告
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # ★★ 模型实际给出了几个编号 / 一共有几个候选。
    #
    #   这两个字段是踩出来的。实测发现同一个模型同一个提示词，有时输出
    #   600+ token 的完整排序，有时只输出 `[1]` 四个 token——而 `_parse_order`
    #   会把没提到的候选按原顺序补齐，于是 `[1]` 的结果 ≈ 原顺序，
    #   **等于重排根本没做事，但 fell_back 是 False，代码报告「成功」**。
    #
    #   也就是说，光看 fell_back 分不出「真的重排了」和「压根没排」。
    #   评测里如果混进一批空转，收益的**幅度**就不可信了
    #   （方向还可信：空转只会稀释收益，不会造出收益）。
    #
    #   所以把「给了几个」暴露出来。先能看见，才谈得上修。
    ranked_count: int = 0
    candidate_count: int = 0

    @property
    def partial(self) -> bool:
        """模型只排了一部分候选 —— 剩下的是按原顺序补的，那部分没有被重排。"""
        return not self.fell_back and self.ranked_count < self.candidate_count


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = get_config()
        if not cfg.agent_configured:
            raise ModuleError("AGENT_API_KEY 未配置，无法重排",
                              code="rerank_not_configured", status=503)
        # ★ 这里就是挂死 13 小时的现场：194 次调用的消融实验，
        #   进程活着、CPU 只用了 14.7 秒，全程在等一个永远不返回的响应。
        _client = OpenAI(base_url=cfg.agent_base_url, api_key=cfg.agent_api_key,
                         timeout=cfg.llm_timeout_seconds,
                         max_retries=cfg.llm_max_retries)
    return _client


def _parse_order(text: str, n: int) -> tuple[list[int] | None, str | None, int]:
    """把模型输出解析成 0-based 下标序列。

    ★ 解析必须 fail-closed：宁可回落到原顺序，也不能把一个残缺的排序当成结果——
      那会让评测测出一个"重排让指标变差了"的假结论，而真实原因是解析出错。

    容错到什么程度是有取舍的：这里只剥掉 ```json 围栏和数组外的杂字，
    **不**尝试从散文里猜编号。猜出来的顺序不可信，回落更诚实。
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[[^\[\]]*\]", cleaned)
    if not match:
        return None, "输出里找不到 JSON 数组", 0
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "JSON 解析失败", 0
    if not isinstance(raw, list):
        return None, "输出不是数组", 0

    seen, order = set(), []
    for item in raw:
        if not isinstance(item, int):
            continue
        idx = item - 1                      # 提示词里是 1-based
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)

    if not order:
        return None, "数组里没有任何合法编号", 0
    ranked = len(order)
    # ★ 模型漏掉的候选按原顺序补在后面，而不是丢掉。
    #   丢掉的话候选数变少，Recall 会凭空下降——那是重排的锅还是解析的锅就分不清了。
    #   但补齐这件事本身要**留痕**（ranked 会一路传到 RerankOutcome.ranked_count），
    #   否则「模型只给了 1 个编号」和「模型给全了」在外面看起来一模一样。
    order += [i for i in range(n) if i not in seen]
    return order, None, ranked


def rerank(query: str, snippets: list[str], *,
           shuffle: bool = True, seed: int = 20260806) -> RerankOutcome:
    """对候选片段重排，返回原下标的新顺序。

    shuffle=False 只在评测里用——用来量位置偏见有多大（见模块 docstring）。
    生产路径永远打乱。
    """
    n = len(snippets)
    if n <= 1:
        return RerankOutcome(order=list(range(n)), fell_back=False,
                             ranked_count=n, candidate_count=n)

    # 打乱输入顺序，消除「模型偏向排在前面的选项」
    presented = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(presented)

    lines = []
    for position, original in enumerate(presented, start=1):
        text = snippets[original][:SNIPPET_CHARS].replace("\n", " ")
        lines.append(f"[{position}] {text}")

    cfg = get_config()
    try:
        response = _get_client().chat.completions.create(
            model=cfg.agent_model,
            messages=[{"role": "user", "content": _PROMPT.format(
                query=query, candidates="\n\n".join(lines))}],
            temperature=0,
        )
    except Exception as exc:                                    # noqa: BLE001
        # ★ 重排失败不能让整个检索失败。它是**增强**，不是必需环节——
        #   降级成"用原来的召回顺序"，用户仍然拿得到结果。
        return RerankOutcome(order=list(range(n)), fell_back=True,
                             reason=f"模型调用失败：{type(exc).__name__}",
                             candidate_count=n)

    usage = response.usage
    order_in_presented, reason, ranked = _parse_order(
        response.choices[0].message.content or "", n)

    if order_in_presented is None:
        return RerankOutcome(order=list(range(n)), fell_back=True, reason=reason,
                             prompt_tokens=usage.prompt_tokens if usage else 0,
                             completion_tokens=usage.completion_tokens if usage else 0,
                             candidate_count=n)

    # 模型给的是"展示位置"的排序，映射回原候选下标
    return RerankOutcome(
        order=[presented[i] for i in order_in_presented],
        fell_back=False,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        ranked_count=ranked,
        candidate_count=n,
    )
