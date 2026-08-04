"""答案溯源检查：模型说的话，有没有出处。

## 为什么要它

这个项目所有评测都在测**检索**——找得准不准、全不全、该不该再找一次。
**从来没测过"模型有没有编"。**

而项目里已经有过真实的软幻觉：

    问「用户研究组是做什么的？」
      它查了花名册（表里只有 部门/小组/人数/组长 四列）
      答：「组长是吴倩，主要负责与用户相关的调研和分析工作」
                            ↑ 表里根本没有"职责"这一列，这半句是编的

而且听起来完全合理——这正是最危险的地方。

## 为什么不用 LLM 当裁判

RAGAS 那套（Faithfulness / Answer Relevancy）都要另一个模型来判。代价：

- **不确定性叠加**：被测系统本身有波动，再叠一个随机裁判，分数变了分不清是谁的原因
- **分数没法调试**：给你 faithfulness = 0.72，你不知道该改哪里
- **进不了 CI**：要花钱、要联网、要几分钟

所以这里做的是**确定性的字符串溯源**：把答案里的数字和标识符抽出来，
逐个去检索到的证据里找。找不到的逐条列出来——**不是打一个分，是指出具体哪一句可疑**。

## 它抓得住什么、抓不住什么

抓得住：
  - 编造的编号 / 型号 / 英文缩写（这些不可能是"算出来的"）
  - 编造的数字（金额、人数、天数）

抓不住：
  - 编造的中文描述（「主要负责用户调研」这种）——需要 NER 或语义判断
  - 措辞正确但因果错误的推断

所以它是**下限**不是全貌：**它报出来的一定可疑，它没报的不一定干净。**
这条必须和数字一起说，否则就成了另一种"看起来很科学"的自欺。

## 一个必须处理的边界：算出来的数字

问「公司总共多少人」，答 84。84 不在任何一条证据里——
它是 32+14+18+9+11 算出来的，**完全正确**。

所以未溯源 ≠ 幻觉。本模块只负责**列出**未溯源的数字，
由调用方按题型区分（聚合类问题本来就该有计算结果）。
"""

from __future__ import annotations

import re

# 引用标记 [1] [12]，要在抽数字之前去掉，否则会把编号当成事实数字
_CITATION = re.compile(r"\[\d+\]")
# markdown 强调符号，会粘在数字上（**8 人**）
_MARKDOWN = re.compile(r"[*`_#>|]")
# 数字：整数、小数、带千分位。不含前后的单位
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# 标识符：字母数字复合串（PRJ-2026-0142 / X7-Pro / SLA / v2.3.1）
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+|[A-Z]{2,}")

# 这些数字不值得追溯：太常见，出现在任何文本里都不奇怪
_TRIVIAL = {"0", "1", "2", "3", "4", "5"}


def _clean(text: str) -> str:
    return _MARKDOWN.sub("", _CITATION.sub(" ", text))


def normalize_number(raw: str) -> str:
    """去掉千分位，让 3,053,000 和 3053000 可比。"""
    return raw.replace(",", "")


def extract_numbers(answer: str) -> list[str]:
    """抽出答案里值得追溯的数字。

    先去掉引用标记——否则 `[1]` 里的 1 会被当成事实数字。
    再去掉个位数：0~5 出现在任何文本里都不奇怪，追溯它们只会制造噪声。
    """
    text = _clean(answer)
    # 先把带千分位的整体抓出来，再抓普通数字，避免 3,053,000 被拆成 3/053/000
    grouped = re.findall(r"\d{1,3}(?:,\d{3})+", text)
    text_without_grouped = re.sub(r"\d{1,3}(?:,\d{3})+", " ", text)
    plain = _NUMBER.findall(text_without_grouped)
    seen, out = set(), []
    for raw in [normalize_number(g) for g in grouped] + plain:
        if raw in _TRIVIAL or raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
    return out


def extract_identifiers(answer: str) -> list[str]:
    """抽出编号 / 型号 / 英文缩写。

    ★ 这类词是溯源检查里最硬的部分：它们**不可能是算出来的**，
    要么来自证据，要么是编的。
    """
    text = _clean(answer)
    seen, out = set(), []
    for token in _IDENTIFIER.findall(text):
        upper = token.upper()
        if upper in seen:
            continue
        seen.add(upper)
        out.append(token)
    return out


def check(answer: str, evidence_texts: list[str]) -> dict:
    """逐个检查答案里的数字和标识符能不能在证据里找到。

    evidence_texts 必须包含**两条链路**的证据——文档片段和 SQL 结果都要。
    实测踩过：问「Delta 项目的负责人管理的小组有多少人」，答案里的 8 来自
    表格查询结果，只看文档证据会误判成幻觉。
    """
    haystack = normalize_number(_clean(" \n ".join(evidence_texts))).upper()

    numbers = extract_numbers(answer)
    identifiers = extract_identifiers(answer)

    ungrounded_numbers = [n for n in numbers if n not in haystack]
    ungrounded_identifiers = [i for i in identifiers if i.upper() not in haystack]

    total = len(numbers) + len(identifiers)
    grounded = total - len(ungrounded_numbers) - len(ungrounded_identifiers)

    return {
        "numbers": numbers,
        "identifiers": identifiers,
        "ungrounded_numbers": ungrounded_numbers,
        "ungrounded_identifiers": ungrounded_identifiers,
        "total_claims": total,
        "grounded_claims": grounded,
        # 没有可查的东西时返回 None 而不是 1.0——
        # 「答案里一个数字都没有」和「所有数字都有出处」是两回事，不该混成同一个分
        "grounding_rate": (grounded / total) if total else None,
    }
