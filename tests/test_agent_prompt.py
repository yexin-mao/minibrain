"""system prompt 的组装。不调模型，只检查该有的段落在不在。

这个文件是**防退化**用的：prompt 里每一段都是实验换来的，
但代码上它们只是几行字符串，最容易被"顺手精简"掉。
每一条断言后面都注明了它是哪次实验的产物、删掉会损失什么。
"""

from __future__ import annotations

import pytest

from minibrain import gateway
from minibrain.agent.prompt import system_prompt as _system_prompt
from minibrain.db import vector_db


@pytest.fixture(scope="module")
def prompt(alice, sales_table):
    return _system_prompt(alice)


def test_includes_table_schema(prompt, sales_table):
    """表结构：最早就有的部分。"""
    assert sales_table in prompt
    assert "销售额" in prompt


def test_includes_column_values(prompt):
    """★ 列取值。

    起因 tbl-18：模型写 WHERE "状态" = '驳回'，实际值是 '已驳回'，返回 0。
    删掉这段 → 模型只能猜枚举值（eval/ROUTING.md 第三节）。
    """
    assert "取值" in prompt


def test_includes_authority_rules(prompt):
    """★ 权威来源规则。

    消融实验（eval/PROMPT_ABLATION.md）：光给文档清单只能把目标类别修到 78%，
    加上这段才 100%。结论是「给资料 ≠ 给判断依据」。
    """
    assert "权威来源规则" in prompt
    assert "以文档为准" in prompt
    assert "记账用的辅助行" in prompt


def test_includes_stopping_rules(prompt):
    """★ 停止判定。

    起因：压力测试里「链条到顶」类问题 50% 耗尽 max_steps，
    连续 6~8 次 vector_search 换措辞反复查同一件事（eval/RESULTS.md 探针五）。
    删掉这段 → 多跳·原有那组的耗尽率会从 0% 回到 33%。
    """
    assert "什么时候停止检索" in prompt
    assert "不要换个措辞反复检索同一件事" in prompt
    assert "查到头了" in prompt


def test_includes_lightweight_multihop_protocol(prompt):
    """多跳规划复用现有 tool loop，不引入一次独立 Planner LLM 调用。"""
    assert "多跳问题的检索协议" in prompt
    assert "中间实体 + 下一条缺失关系" in prompt
    assert "只找到中间实体，不等于已经找到最终答案" in prompt
    assert "时间、对象和条件范围" in prompt
    assert "不要擅自把问题扩成" in prompt
    assert "单跳问题查到答案后立即回答" in prompt


def test_says_not_found_is_a_valid_answer(prompt):
    """「查不到」是正确答案而不是失败——这句是防幻觉的关键。"""
    assert "正确答案，不是失败" in prompt


def test_includes_knowledge_domain_without_per_document_catalog(alice):
    """★ 可扩展知识域摘要。

    没有它时「文档·组织事实」类路由只有 1/6，因为模型看得见表有哪些列，
    完全不知道文档里写了什么（eval/ROUTING.md 第一节）。
    """
    # 这里只测 Router 的 O(source 数)摘要，不测 embedding 入库。直接造一个 ready
    # 登记项可避免付费 API，也避免开发机常驻 worker 抢走测试任务导致 teardown 等待。
    source = gateway.call("vector-rag", "ensure_default_source", alice)
    with vector_db() as cur:
        cur.execute(
            """INSERT INTO documents
               (source_id, filename, content, status, char_count, parsed_as, content_hash)
               VALUES (%s, 'dept-tech.md', '# 技术部', 'ready', 5, 'markdown', 'prompt-test')""",
            (source["id"],),
        )
    text = _system_prompt(alice)
    assert "可检索的知识域/source" in text
    assert f"user/{alice.username}" in text
    assert "1 篇" in text
    assert "dept-tech.md" not in text


def test_declares_current_user(prompt, alice):
    """身份要写进 prompt：模型据此知道自己看到的是过滤后的内容。"""
    assert alice.username in prompt
