"""把两条链路每一步的中间产物打印出来。

这个项目的立意是"两条范式不同的链路不能合并"。README 用文字说了，
评测用数字证了，但**光看文字和数字，很难对"它们到底哪里不一样"有直觉**。

这个脚本不测任何东西，只做一件事：把上传 → 入库 → 检索三步里
每一步的中间产物原样打印出来。跑一遍，差别是看得见的：

  文档链路：把文件**打碎**成片段，每片换成 1024 个小数，检索时比谁离得近
  表格链路：把文件**还原**成一张真表，检索时让数据库算一遍

最后一段演示"把表格传进文档链路会怎样"——这是新人最容易犯的错，
也是整个项目论点最直观的证据。

跑法：uv run --no-sync python scripts/explain.py
没配 API key 也能跑：文档链路会跳过，表格链路用预置 SQL 代替模型生成。
"""

from __future__ import annotations

import csv
import io
import math
import secrets
import sys

sys.path.insert(0, "src")

from psycopg import sql as pgsql                                  # noqa: E402

from minibrain.config import get_config                           # noqa: E402
from minibrain.db import close_all, table_db                      # noqa: E402
from minibrain.modules.vector_rag.chunking import split_text      # noqa: E402

DOC = """# 差旅住宿标准

一线城市每晚不超过 600 元，二线城市每晚不超过 400 元，其他城市每晚不超过 300 元。

# 年假制度

入职满 1 年享有 5 天年假，满 3 年享有 10 天，满 5 年享有 15 天。

# 报销流程

发票需在消费后 30 天内提交。超过 30 天的，需由所在部门负责人出具书面说明。
"""

CSV_TEXT = """区域,月份,销售额
华东,1,182000
华东,2,196500
华东,3,221000
华北,1,128000
华北,2,135500
华北,3,149000
"""

QUESTION_DOC = "住酒店一晚能报多少钱"
QUESTION_TBL = "华东区的销售额合计是多少？"

# 刻意调小：项目默认 800 字，那个尺寸下这篇短文只会切成 1 段，看不出切分的效果。
# 副作用是边界被切得很碎——这恰好演示了为什么检索要取前 N 名而不是第 1 名。
DEMO_CHUNK_SIZE = 60
DEMO_OVERLAP = 10


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / (norm or 1.0)


# ---------------------------------------------------------------- 链路一

def explain_vector_chain() -> None:
    rule("链路一：文档 —— 把文件打碎成片段，检索时找『最像』的片段")

    print("\n【第 1 步 上传】原文整篇存进数据库，一个字都不动，然后立刻返回")
    print(f"  存了 {len(DOC)} 个字符")
    print("  为什么不当场处理完？下一步要调模型，几十份文件要好几分钟，网页不能卡在那儿等。")

    print("\n【第 2 步 入库 a：切分】按段落切成小块，超长的段落硬切并留重叠")
    chunks = split_text(DOC, DEMO_CHUNK_SIZE, DEMO_OVERLAP)
    for i, c in enumerate(chunks):
        print(f"  片段{i}: {c[:38].replace(chr(10), ' / ')}...")
    print("  为什么要切？后面要把片段喂给模型，不能把整个知识库塞给它。")

    if not get_config().embedding_configured:
        print("\n【第 3、4 步】需要 EMBEDDING_API_KEY，已跳过。")
        print("  这两步是：每个片段调模型换成一串数字（向量），检索时比谁离得近。")
        return

    from minibrain.modules.vector_rag.embeddings import embed_query, embed_texts

    print("\n【第 3 步 入库 b：转成向量】每个片段送去模型，换回一串数字")
    vectors = embed_texts(chunks)
    print(f"  每个片段 → {len(vectors[0])} 个小数。片段0 的前 6 个：")
    print(f"  {[round(x, 4) for x in vectors[0][:6]]} …（后面还有 {len(vectors[0]) - 6} 个）")
    print("  把它想成这句话在『意思空间』里的坐标——意思相近的两句话，坐标就相近。")
    print("  「住酒店能报多少」和「差旅住宿标准」没一个字重合，但坐标是挨着的。")

    print("\n【第 4 步 检索】问题也转成同样的坐标，然后比谁离得近")
    print(f"  问题：{QUESTION_DOC}")
    qv = embed_query(QUESTION_DOC)
    print(f"  问题 → {[round(x, 4) for x in qv[:6]]} …\n")

    scored = sorted(
        ((cosine(qv, v), i, c) for i, (v, c) in enumerate(zip(vectors, chunks))),
        reverse=True,
    )
    print("  相似度打分（越接近 1 越像）：")
    for score, i, c in scored:
        print(f"    {score:.4f}  片段{i}: {c[:34].replace(chr(10), ' / ')}...")

    top_idx = scored[0][1]
    print(f"\n  → 取前 N 名的原文交给模型去组织回答。")
    if top_idx != 0:
        print(f"  ⚠ 注意：排第一的是片段{top_idx}，但它是被切残的——完整答案在片段0 里。")
        print("    这就是为什么实际取前 5 名而不是第 1 名：单靠第一名很可能拿到半截答案。")


# ---------------------------------------------------------------- 链路二

def explain_table_chain() -> None:
    rule("链路二：表格 —— 把文件还原成一张真表，检索时执行计算")

    table_name = f"t_{secrets.token_hex(4)}"
    print("\n【第 1 步 上传】CSV 原文整篇存进数据库，同时生成一个随机表名")
    print(f"  表名 = {table_name}")
    print("  为什么随机生成？绝不能让用户输入变成表名——那是 SQL 注入的入口。")

    print("\n【第 2 步 入库 a：解析】读成一行一行")
    rows = list(csv.reader(io.StringIO(CSV_TEXT)))
    header, body = rows[0], rows[1:]
    print(f"  表头：{header}")
    print(f"  数据：{len(body)} 行，第一行 {body[0]}")

    types = [
        "numeric" if all(r[i].replace(".", "").isdigit() for r in body) else "text"
        for i in range(len(header))
    ]

    print("\n【第 3 步 入库 b：建一张真的数据库表，把行灌进去】")
    print(f"  推断列类型：{dict(zip(header, types))}")
    cols = ", ".join(f'"{h}" {t}' for h, t in zip(header, types))
    print(f"  CREATE TABLE {table_name} ({cols})")
    print(f"  INSERT INTO {table_name} VALUES … × {len(body)} 行")
    print("  ★ 这一步跟文档链路彻底分道扬镳：没有切分，没有向量，没有相似度。")
    print("    文档链路把文件打碎，表格链路把文件原样重建——一行不少，一列不缺。")

    ident = pgsql.Identifier(table_name)
    try:
        with table_db() as cur:
            cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(ident))
            cur.execute(pgsql.SQL("CREATE TABLE {} ({})").format(
                ident,
                pgsql.SQL(", ").join(
                    pgsql.SQL("{} {}").format(pgsql.Identifier(h), pgsql.SQL(t))
                    for h, t in zip(header, types)
                ),
            ))
            cur.executemany(
                pgsql.SQL("INSERT INTO {} VALUES ({})").format(
                    ident, pgsql.SQL(", ").join(pgsql.Placeholder() * len(header))
                ),
                body,
            )

        print("\n【第 4 步 检索】把表结构告诉模型，让它写一条 SQL，然后数据库来算")
        schema_text = f"表 {table_name}，列：{cols}"
        print(f"  问题：{QUESTION_TBL}")
        print(f"  告诉模型：{schema_text}\n")

        sql_text = _generate_sql(schema_text)
        with table_db() as cur:
            cur.execute(f'SELECT * FROM ({sql_text.rstrip(";")}) AS _guarded LIMIT 200')
            result = cur.fetchall()

        print(f"  数据库算出来：{result}")
        print(f"  → 这个数字是全部 {len(body)} 行里符合条件的行加起来的，一行都不会漏。")
        print("  ★ 模型在这条链路里只负责写查询语句，不负责算数。")

    finally:
        with table_db() as cur:
            cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(ident))


def _generate_sql(schema_text: str) -> str:
    cfg = get_config()
    if not cfg.agent_configured:
        fallback = f'SELECT sum("销售额") FROM {schema_text.split("，")[0][2:]} WHERE "区域" = \'华东\''
        print(f"  （未配 AGENT_API_KEY，用预置 SQL 代替模型生成）")
        print(f"  SQL：{fallback}\n")
        return fallback

    from openai import OpenAI

    client = OpenAI(base_url=cfg.agent_base_url, api_key=cfg.agent_api_key)
    response = client.chat.completions.create(
        model=cfg.agent_model,
        temperature=0,
        messages=[
            {"role": "system", "content":
                f"你只输出一条 SQL，不要解释，不要 markdown 代码块。"
                f"标识符用双引号。可用表：{schema_text}"},
            {"role": "user", "content": QUESTION_TBL},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    text = text.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
    print(f"  模型写出来的 SQL：{text}\n")
    return text


# ---------------------------------------------------------------- 对比

def explain_wrong_chain() -> None:
    rule("如果走错链路会怎样（这是新人最容易犯的错）")

    rows = len(list(csv.reader(io.StringIO(CSV_TEXT)))) - 1
    csv_chunks = split_text(CSV_TEXT, DEMO_CHUNK_SIZE, DEMO_OVERLAP)

    print(f"\n  把这张 {rows} 行的表当成文档传进去，会被切成 {len(csv_chunks)} 个片段：")
    for i, c in enumerate(csv_chunks):
        print(f"    片段{i}: {c[:45]!r}")

    print(f"\n  看片段1 的开头——有半行数据被从中间切断了，只剩一个孤零零的数字。")
    print(f"  检索时只会取回最像的一两个片段，模型只看到部分行，加出来的合计必然是错的。")
    print(f"  而且它会算得很认真、答得很像样，你看不出错。")
    print(f"\n  这就是 eval/RESULTS.md 里那个『答 73 人，真值 84 人』的失效模式。")
    print(f"  ★ 一张报表切碎再向量召回回来，算不出正确的合计——这是两条链路不能合并的理由。")


def main() -> int:
    explain_vector_chain()
    explain_table_chain()
    explain_wrong_chain()
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
