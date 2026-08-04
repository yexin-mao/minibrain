"""表格链路：入库、类型推断、精确计算。

这条链路存在的理由就是最后那个测试——一张报表切碎再向量召回回来，
算不出正确的合计。sum() 会算对，且不会漏行。
"""

from __future__ import annotations

from minibrain import gateway


def test_csv_reaches_ready(alice, sales_table):
    row = next(d for d in gateway.call("table-rag", "list_datasets", alice)
               if d["table_name"] == sales_table)
    assert row["status"] == "ready"
    assert row["error"] is None


def test_row_count_is_correct(alice, sales_table):
    row = next(d for d in gateway.call("table-rag", "list_datasets", alice)
               if d["table_name"] == sales_table)
    assert row["row_count"] == 4


def test_numeric_column_is_detected(alice, sales_table):
    row = next(d for d in gateway.call("table-rag", "list_datasets", alice)
               if d["table_name"] == sales_table)
    types = {c["name"]: c["type"] for c in row["columns"]}
    assert types["销售额"] == "numeric"
    assert types["地区"] == "text"


def test_chinese_column_names_are_preserved(alice, sales_table):
    """列名保留中文原文，代价是 LLM 必须给标识符加双引号（system prompt 里有要求）。"""
    row = next(d for d in gateway.call("table-rag", "list_datasets", alice)
               if d["table_name"] == sales_table)
    assert [c["name"] for c in row["columns"]] == ["地区", "产品", "销售额"]


def test_physical_table_name_is_module_generated(sales_table):
    """表名由模块生成，用户输入永远不进标识符。"""
    assert sales_table.startswith("t_")
    assert len(sales_table) == 10          # t_ + 8 位 hex


def test_group_by_sum_is_exact(alice, sales_table):
    """★ 这就是两条链路不能合并的理由：聚合必须精确，不能靠召回片段猜。"""
    result = gateway.call(
        "table-rag", "run_query", alice,
        f'SELECT "地区", sum("销售额") AS 合计 FROM {sales_table} '
        f'GROUP BY "地区" ORDER BY 合计 DESC',
    )
    assert result["row_count"] == 2
    assert float(result["rows"][0]["合计"]) == 2000      # 华东 1200 + 800
    assert float(result["rows"][1]["合计"]) == 800       # 华北 500 + 300


def test_plain_select_still_works(alice, sales_table):
    """护栏不能把正常查询也拦掉。"""
    result = gateway.call("table-rag", "run_query", alice, f"SELECT * FROM {sales_table}")
    assert result["row_count"] == 4


def test_describe_schema_lists_visible_tables(alice, sales_table):
    """这段文本会被注入 Agent 的 system prompt，是路由质量的直接输入。"""
    schema = gateway.call("table-rag", "describe_schema", alice)
    assert sales_table in schema
    assert "销售额" in schema


# ---------------------------------------------------------------- 低基数列的取值注入
#
# 起因：tbl-18「有几笔报销被驳回」，模型写 WHERE "状态" = '驳回'，
# 实际值是 '已驳回'，返回 0。根因是 schema 只给列名和类型，模型只能猜枚举值。
# text-to-SQL 领域称之为 value linking。

def test_schema_includes_low_cardinality_values(alice):
    csv = "地区,状态,金额\n华东,已通过,100\n华北,已驳回,200\n华东,已通过,300\n"
    ds_id = gateway.call("table-rag", "upload_csv", alice, None, "status.csv", csv.encode())
    gateway.process("table-rag", ds_id)

    schema = gateway.call("table-rag", "describe_schema", alice)
    assert "已驳回" in schema, "低基数文本列的取值必须注入，否则模型只能猜"
    assert "已通过" in schema


def test_schema_omits_high_cardinality_values(alice):
    """★ 高基数列不能注入——列出来没意义，还挤占上下文。"""
    rows = "\n".join(f"BX{i:08d},{i}" for i in range(40))
    csv = f"单号,金额\n{rows}\n"
    ds_id = gateway.call("table-rag", "upload_csv", alice, None, "manyids.csv", csv.encode())
    gateway.process("table-rag", ds_id)

    table = next(d["table_name"] for d in gateway.call("table-rag", "list_datasets", alice)
                 if d["filename"] == "manyids.csv")
    block = next(b for b in gateway.call("table-rag", "describe_schema", alice).split("表 ")
                 if b.startswith(table))
    assert "BX00000000" not in block
    assert "全部取值" not in block


def test_schema_omits_long_free_text_values(alice):
    """自由文本列即使取值少也不注入——那不是枚举，是正文。"""
    long_a, long_b = "这是一段很长的自由文本内容" * 4, "另一段同样很长的自由文本" * 4
    csv = f"备注,金额\n{long_a},1\n{long_b},2\n"
    ds_id = gateway.call("table-rag", "upload_csv", alice, None, "notes.csv", csv.encode())
    gateway.process("table-rag", ds_id)

    table = next(d["table_name"] for d in gateway.call("table-rag", "list_datasets", alice)
                 if d["filename"] == "notes.csv")
    block = next(b for b in gateway.call("table-rag", "describe_schema", alice).split("表 ")
                 if b.startswith(table))
    assert "全部取值" not in block


def test_schema_values_respect_permissions(alice, bob):
    """取值也是数据。别人的表的取值绝不能出现在我的 prompt 里。"""
    csv = "地区,机密标记\n华东,绝密项目代号A\n"
    ds_id = gateway.call("table-rag", "upload_csv", alice, None, "confidential.csv", csv.encode())
    gateway.process("table-rag", ds_id)
    assert "绝密项目代号A" not in gateway.call("table-rag", "describe_schema", bob)
