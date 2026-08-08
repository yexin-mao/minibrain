"""Agent 的返回类型。**框架版和手写版共用**，所以单独放一个文件。

## 为什么必须共用

`AnswerResult` / `ToolCallTrace` 的形状决定了三件事能不能对齐：

1. **Web 的证据面板** —— `_answer.html` 直接渲染 `result.evidence` 和 `result.trace`
2. **轨迹评测** —— `scripts/eval_trajectory.py` 的全部指标
   （边际证据收益、冗余检索率、轮数分布）都建立在 `ToolCallTrace` 上
3. **两版可比** —— 同一套评测脚本不改一行就能同时量 LangGraph 版和手写版

如果换框架时顺手改了这个形状，前面所有的评测结论都得重跑。
所以它被单独抽出来，谁也不许动。

★ `graph.py` 里有个 `_to_trace()` 专门把 LangGraph 的 message 列表
  还原成这个形状。那不是多余的适配代码，它是"可比"的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import Evidence


@dataclass
class ToolCallTrace:
    """一次工具调用的留痕。

    三个字段都不是可选的：
    - `name` 用来算工具选择准确率
    - `arguments` 用来算冗余检索率（同一个查询查了两遍）
    - `result_preview` 用来算边际证据收益（这轮带来多少新证据）

    ★ `result_preview` 截断到 300 字，所以边际证据收益是**低估值**。
      低估比高估安全：如果连低估都显示第 2 轮有收益，那就是真的有。
    """

    name: str
    arguments: str
    result_preview: str


@dataclass
class AnswerResult:
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    trace: list[ToolCallTrace] = field(default_factory=list)
