"""冒烟测试：不需要任何 API key，验证边界是否真的成立。

覆盖的是最值得钉死的几条：权限隔离、只读护栏、状态机。
检索质量之类的东西不在这里测 —— 那需要评测集，不是冒烟能解决的。

跑法：uv run python scripts/smoke.py
"""

from __future__ import annotations

import sys
import uuid

sys.path.insert(0, "src")

from minibrain import gateway, identity                       # noqa: E402
from minibrain.config import get_config                        # noqa: E402
from minibrain.contracts import ModuleError, PermissionDenied  # noqa: E402
from minibrain.db import close_all                             # noqa: E402
from minibrain.scripts_purge import purge_user                 # noqa: E402

CSV = """地区,产品,销售额
华东,A,1200
华东,B,800
华北,A,500
华北,B,300
"""

passed = failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  ok   {label}")
    else:
        failed += 1
        print(f"  FAIL {label} {detail}")


def main() -> int:
    tag = uuid.uuid4().hex[:6]
    alice = identity.create_user(f"smoke_a_{tag}", "pw123456", is_admin=False)
    bob = identity.create_user(f"smoke_b_{tag}", "pw123456", is_admin=False)
    admin = identity.create_user(f"smoke_admin_{tag}", "pw123456", is_admin=True)

    print("\n[1] 身份层")
    token = identity.authenticate(f"smoke_a_{tag}", "pw123456")
    check("登录签发 token", bool(token))
    check("token 能还原成 UserContext", identity.resolve_session(token).user_id == alice.user_id)
    check("错误 token 得到 None", identity.resolve_session("garbage") is None)
    try:
        identity.authenticate(f"smoke_a_{tag}", "wrong")
        check("错误密码被拒", False)
    except ModuleError:
        check("错误密码被拒", True)

    print("\n[2] 表格链路入库")
    dataset_id = gateway.call("table-rag", "upload_csv", alice, None, "sales.csv", CSV.encode())
    gateway.process("table-rag", dataset_id)
    datasets = gateway.call("table-rag", "list_datasets", alice)
    mine = [d for d in datasets if str(d["id"]) == dataset_id][0]
    check("状态到 ready", mine["status"] == "ready", f'实际 {mine["status"]} {mine["error"]}')
    check("行数正确", mine["row_count"] == 4, f'实际 {mine["row_count"]}')
    check("数值列被识别为 numeric",
          any(c["name"] == "销售额" and c["type"] == "numeric" for c in mine["columns"]),
          str(mine["columns"]))
    table = mine["table_name"]

    print("\n[3] 精确计算（这是向量检索做不到的）")
    result = gateway.call(
        "table-rag", "run_query", alice,
        f'SELECT "地区", sum("销售额") AS 合计 FROM {table} GROUP BY "地区" ORDER BY 合计 DESC',
    )
    check("分组求和返回 2 行", result["row_count"] == 2, str(result["rows"]))
    check("华东合计 = 2000", float(result["rows"][0]["合计"]) == 2000, str(result["rows"]))

    print("\n[4] 权限隔离")
    check("bob 看不到 alice 的私有表", table not in
          {d["table_name"] for d in gateway.call("table-rag", "visible_tables", bob)})
    try:
        gateway.call("table-rag", "run_query", bob, f"SELECT * FROM {table}")
        check("bob 直接查表名被拒", False, "越权成功了")
    except PermissionDenied:
        check("bob 直接查表名被拒", True)
    check("管理员看得到", table in
          {d["table_name"] for d in gateway.call("table-rag", "visible_tables", admin)})

    print("\n[5] 只读护栏")
    for label, stmt in [
        ("DELETE 被拒", f"DELETE FROM {table}"),
        ("UPDATE 被拒", f'UPDATE {table} SET "销售额" = 0'),
        ("多语句被拒", f"SELECT 1; DROP TABLE {table}"),
        ("跨 schema 读身份表被拒", "SELECT * FROM identity.users"),
        ("未登记的表被拒", "SELECT * FROM pg_catalog.pg_tables"),
    ]:
        try:
            gateway.call("table-rag", "run_query", alice, stmt)
            check(label, False, "居然执行成功了")
        except ModuleError:
            check(label, True)

    result = gateway.call("table-rag", "run_query", alice, f"SELECT * FROM {table}")
    check("正常 SELECT 仍可用", result["row_count"] == 4)

    print("\n[6] 向量链路状态机")
    doc_id = gateway.call("vector-rag", "upload_document", alice, None, "note.md", "# 标题\n\n正文内容。")
    docs = gateway.call("vector-rag", "list_documents", alice)
    check("上传后立刻返回，状态是 uploaded",
          [d for d in docs if str(d["id"]) == doc_id][0]["status"] == "uploaded")

    gateway.process("vector-rag", doc_id)
    docs = gateway.call("vector-rag", "list_documents", alice)
    final = [d for d in docs if str(d["id"]) == doc_id][0]

    # 两条路径都要钉死：配了 key 就该 ready，没配就该 failed 并写明原因。
    # 关键是任何一边都不许出现"状态是 ready 但其实没索引成功"。
    if get_config().embedding_configured:
        check("配了 key：状态到 ready 且有片段",
              final["status"] == "ready" and final["chunk_count"] > 0,
              f'实际 {final["status"]} chunks={final["chunk_count"]} {final["error"]}')
    else:
        check("没配 key：落 failed 而不是假装 ready",
              final["status"] == "failed" and final["error"] is not None,
              f'实际 {final["status"]}')

    check("bob 看不到 alice 的文档",
          doc_id not in {str(d["id"]) for d in gateway.call("vector-rag", "list_documents", bob)})

    print("\n[7] 自清理")
    # 测试建的是真实用户和真实数据。跑完必须收拾干净，
    # 否则跑几次之后管理员的知识列表就全是垃圾，没人敢再跑。
    for u in (alice, bob, admin):
        purge_user(u)
    remaining = {x.username for x in identity.list_users()}
    check("测试用户已全部清除", not any(u.username in remaining for u in (alice, bob, admin)))
    check("物理表已 DROP", not _orphan_tables(tag))

    print(f"\n{'=' * 46}\n通过 {passed}，失败 {failed}")
    return 1 if failed else 0


def _orphan_tables(tag: str) -> list[str]:
    """确认 DROP 真的执行了：物理表不在外键图里，只靠 cascade 是删不掉的。"""
    from minibrain.db import table_db

    with table_db() as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'mod_table' AND tablename LIKE 't\\_%'"
        )
        physical = {r["tablename"] for r in cur.fetchall()}
        cur.execute("SELECT table_name FROM datasets")
        registered = {r["table_name"] for r in cur.fetchall()}
    return sorted(physical - registered)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
