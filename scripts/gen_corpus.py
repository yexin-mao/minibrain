"""生成扩充语料。

## 为什么要扩

原语料 15 篇、1951 字符。在这个规模上向量检索的指标已经顶到天花板：

    MRR = 1.000    Hit Rate@k = 1.000    Recall@10 = 1.000

**没有提升空间，就没法证明任何检索侧改进有效。** 混合检索、rerank、
分层索引、语义路由——做了都会得到"指标没变"，而那说不清是方法没用
还是根本测不出来。

## 扩语料不是"多写点文档"

随便多写 85 篇同类型的短文，只会把天花板顶得更死一点。要能测出差别，
必须**刻意植入向量检索的盲区**——那些语义相似度天然抓不住的东西：

| 盲区类型 | 例子 | 为什么向量检索弱 |
|---|---|---|
| 工单/项目编号 | PRJ-2026-0142 | 转成向量就是一串无语义字符，被"平均"掉 |
| 产品型号/版本号 | X7-Pro、v2.3.1 | 同上，且型号之间语义几乎无差别 |
| 英文缩写 | SLA、RTO、P0 | 中文语料里这类词的向量表示很弱 |
| 罕见人名 | 郗昭、乜文渊 | 生僻字在训练语料里出现少 |

植入之后跑一遍探针，如果这类查询的 Recall 明显低于普通查询，
**就直接证明了必须做混合检索**——而不是因为教程里都这么写。

## 一致性约束

新语料必须和原有 15 篇 + 3 张 CSV 对得上，否则评测集自己先矛盾了：

- 部门人数 32/14/18/9/11 = 84，**不能被新文档改变**
- 因此罕见人名一律给**编制外**角色（供应商对接人、外部顾问、候选人）
- 已有人员（张敏、刘洋等 16 人）只能引用，不能改变其归属

跑法：uv run --no-sync python scripts/gen_corpus.py
      uv run --no-sync python scripts/gen_corpus.py --clean   # 先删掉上次生成的
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
BLINDSPOTS = ROOT / "eval" / "blindspots.json"

# 原始 15 篇，永远不动——它们是既有探针的答案所在
ORIGINAL = {
    "dept-finance.md", "dept-hr.md", "dept-market.md", "dept-product.md",
    "dept-tech.md", "intern-liwei.md", "policy-leave.md", "policy-reimburse.md",
    "policy-travel.md", "project-alpha.md", "team-algo.md", "team-backend.md",
    "team-design.md", "team-frontend.md", "team-research.md",
}

# 已在编制内的人（原语料 + CSV），只能引用不能改动归属
STAFF = {
    "李伟": ("技术部", "负责人"), "张敏": ("技术部", "后端组组长"),
    "刘洋": ("技术部", "前端组组长"), "陈刚": ("技术部", "算法组组长"),
    "王芳": ("产品部", "负责人"), "胡静": ("产品部", "产品设计组组长"),
    "吴倩": ("产品部", "用户研究组组长"), "赵磊": ("市场部", "负责人"),
    "冯磊": ("市场部", "品牌组组长"), "许娜": ("市场部", "渠道组组长"),
    "孙红": ("人力资源部", "负责人"), "何静": ("人力资源部", "招聘组组长"),
    "邓超": ("人力资源部", "薪酬组组长"), "周涛": ("财务部", "负责人"),
    "曹敏": ("财务部", "会计组组长"), "袁伟": ("财务部", "审计组组长"),
}

# ---------------------------------------------------------------- 盲区词表
#
# 每一项都会被记录进 blindspots.json：植入在哪篇、属于哪类、
# 以及一句"正常人会怎么问它"——探针题直接从这里生成。

RARE_NAMES = [
    ("郗昭", "云启数据"), ("乜文渊", "恒信咨询"), ("邝允中", "锐锋科技"),
    ("笪立诚", "中衡审计"), ("逯明修", "博远设计"), ("岑柏舟", "天元法务"),
    ("蔺孝廉", "启明云"), ("阚彦博", "泰盛物流"),
]

PRODUCTS = [
    ("X7-Pro", "企业版核心引擎", "v2.3.1"),
    ("X7-Lite", "企业版轻量引擎", "v2.3.1"),
    ("S3-Max", "标准版数据采集器", "v1.8.4"),
    ("S3-Mini", "标准版边缘节点", "v1.8.4"),
    ("K2-Air", "移动端 SDK", "v0.9.12"),
    ("K2-Ground", "桌面端 SDK", "v0.9.12"),
    ("R5-Nova", "报表渲染服务", "v4.1.0"),
    ("R5-Quasar", "报表导出服务", "v4.1.2"),
]

ACRONYMS = [
    ("SLA", "服务等级协议", "线上核心接口的 SLA 承诺为 99.95% 可用性，月度不可用时长不超过 21 分钟"),
    ("RTO", "恢复时间目标", "一级故障的 RTO 为 30 分钟，即从故障确认到服务恢复不得超过 30 分钟"),
    ("RPO", "恢复点目标", "数据库 RPO 为 5 分钟，即最多容忍丢失 5 分钟内的写入"),
    ("QPS", "每秒查询数", "报表服务单实例峰值 QPS 为 1200，超过后自动横向扩容"),
    ("OKR", "目标与关键结果", "OKR 每季度制定一次，公司级 OKR 由管理层在季度首月第一周对齐"),
    ("P0", "最高优先级故障", "P0 故障指影响全部用户的服务中断，必须 15 分钟内响应"),
    ("P1", "高优先级故障", "P1 故障指影响部分用户核心功能，须 1 小时内响应"),
    ("CAC", "获客成本", "渠道组考核的 CAC 上限为 380 元，超出需提交专项说明"),
    ("MRR", "月度经常性收入", "企业版 MRR 是市场部的首要north star 指标"),
    ("DAU", "日活跃用户数", "移动端 DAU 统计口径为当日启动过应用的去重设备数"),
]

PROJECTS = [
    ("PRJ-2026-0142", "Beta 项目", "算法组", "陈刚", "推荐策略重构", "2026 年 5 月至 10 月"),
    ("PRJ-2026-0155", "Gamma 项目", "前端组", "刘洋", "管理后台改版", "2026 年 6 月至 9 月"),
    ("PRJ-2026-0163", "Delta 项目", "后端组", "张敏", "交易链路拆分", "2026 年 3 月至 8 月"),
    ("PRJ-2026-0178", "Epsilon 项目", "用户研究组", "吴倩", "企业客户访谈计划", "2026 年 7 月至 11 月"),
    ("PRJ-2026-0191", "Zeta 项目", "品牌组", "冯磊", "品牌视觉升级", "2026 年 4 月至 7 月"),
    ("PRJ-2026-0204", "Eta 项目", "会计组", "曹敏", "费用系统对接", "2026 年 8 月至 12 月"),
    ("PRJ-2026-0217", "Theta 项目", "产品设计组", "胡静", "设计规范统一", "2026 年 5 月至 8 月"),
    ("PRJ-2026-0223", "Iota 项目", "招聘组", "何静", "校园招聘扩编", "2026 年 9 月至 12 月"),
]

TICKETS = [
    ("TICKET-88231", "报表导出超时", "R5-Quasar", "P1", "张敏"),
    ("TICKET-88407", "移动端崩溃率上升", "K2-Air", "P0", "刘洋"),
    ("TICKET-88512", "推荐结果重复", "X7-Pro", "P1", "陈刚"),
    ("TICKET-88690", "边缘节点心跳丢失", "S3-Mini", "P1", "张敏"),
    ("TICKET-88745", "报销单据无法上传", "R5-Nova", "P1", "曹敏"),
    ("TICKET-88823", "数据采集延迟", "S3-Max", "P0", "陈刚"),
]

MEETINGS = [
    ("MTG-20260415-03", "2026 年 4 月 15 日", "季度技术评审", "李伟",
     "评审 Delta 项目的交易链路拆分方案，决定分三期上线，第一期不含结算模块"),
    ("MTG-20260422-01", "2026 年 4 月 22 日", "产品需求对齐", "王芳",
     "确认 Gamma 项目管理后台改版范围，砍掉自定义仪表盘，保留导出能力"),
    ("MTG-20260508-02", "2026 年 5 月 8 日", "市场季度复盘", "赵磊",
     "复盘华东区渠道投放，决定把 CAC 考核上限从 420 元下调到 380 元"),
    ("MTG-20260519-04", "2026 年 5 月 19 日", "线上故障复盘", "张敏",
     "复盘 TICKET-88407 移动端崩溃，根因是 K2-Air v0.9.11 的内存泄漏，已在 v0.9.12 修复"),
    ("MTG-20260603-01", "2026 年 6 月 3 日", "人力季度例会", "孙红",
     "通过 Iota 项目校园招聘扩编方案，技术部新增 6 个校招名额"),
    ("MTG-20260617-02", "2026 年 6 月 17 日", "财务预算评审", "周涛",
     "评审 Eta 项目费用系统对接预算，批准 48 万元，分两期拨付"),
    ("MTG-20260701-05", "2026 年 7 月 1 日", "架构治理周会", "李伟",
     "确定 X7-Pro 与 X7-Lite 共用同一套配置中心，配置项差异通过 profile 隔离"),
    ("MTG-20260715-01", "2026 年 7 月 15 日", "设计规范评审", "胡静",
     "Theta 项目设计规范统一方案通过，2026 年 9 月起所有新页面强制遵循"),
]

POLICIES = [
    ("policy-overtime", "加班与调休",
     "工作日加班满 2 小时可申请 0.5 天调休，满 4 小时可申请 1 天。"
     "调休需在加班发生后 90 天内使用，逾期作废。法定节假日加班按 3 倍工资结算，不可转调休。"),
    ("policy-performance", "绩效考核",
     "绩效考核每半年一次，分为 S/A/B/C 四档。S 档比例不超过团队人数的 10%，"
     "C 档为不合格，连续两次 C 档启动改进计划。考核结果与年终奖系数直接挂钩。"),
    ("policy-recruit", "招聘流程",
     "招聘需求由用人部门提交，经部门负责人与人力资源部双签后生效。"
     "技术岗位固定四轮面试：初筛、技术一面、技术二面、总监面。offer 审批由孙红最终签发。"),
    ("policy-training", "培训与发展",
     "每位员工每年享有 40 小时带薪培训时长。外部培训费用单次超过 3000 元的，"
     "需提前 15 个工作日提交申请。培训结束后 30 天内需提交学习总结。"),
    ("policy-device", "设备申领",
     "新员工入职当天由 IT 统一配发笔记本电脑，标准配置为 16 英寸、32GB 内存。"
     "设备更换周期为 3 年，提前更换需说明原因。离职时设备须当日归还。"),
    ("policy-confidential", "保密制度",
     "客户名单、财务数据、未发布的产品规划均属于机密信息。"
     "机密文件不得存放在个人设备或公有云盘。违反保密制度的，视情节给予警告直至解除劳动合同。"),
    ("policy-resign", "离职流程",
     "正式员工离职需提前 30 天提交书面申请，试用期员工提前 3 天。"
     "离职交接清单须由直属上级签字确认。工资结算在离职当月的下一个发薪日完成。"),
    ("policy-trip-approval", "出差审批",
     "省内出差由部门负责人审批，跨省出差需再经分管副总审批。"
     "单次出差预算超过 8000 元的，须提前提交行程与预算说明。出差期间的住宿标准见差旅住宿标准。"),
    ("policy-attendance", "考勤制度",
     "实行弹性工作制，核心工作时间为 10:00 至 16:00。月度迟到累计超过 3 次的，"
     "由直属上级提醒；超过 6 次的计入当期绩效。全月无迟到早退可获得全勤奖 200 元。"),
    ("policy-referral", "内部推荐",
     "内推候选人入职并通过试用期后，推荐人可获得奖金。普通岗位 3000 元，"
     "技术岗位 6000 元，总监级岗位 15000 元。奖金在候选人转正后的下一个发薪日发放。"),
    ("policy-checkup", "健康体检",
     "公司每年组织一次员工体检，入职满 6 个月的员工可参加。"
     "体检机构由人力资源部统一选定，员工亦可自行选择机构后凭发票报销，上限 800 元。"),
    ("policy-team-building", "团建经费",
     "团建经费标准为每人每季度 300 元，由部门统一安排。"
     "经费不可跨季度累计，不可折现。单次团建人数少于部门人数一半的，费用不予报销。"),
    ("policy-equity", "股权激励",
     "股权激励面向入职满 1 年且绩效为 A 档及以上的员工。授予后分四年归属，"
     "第一年归属 25%，其后每季度归属 6.25%。离职时未归属部分自动失效。"),
    ("policy-supplier", "供应商管理",
     "新增供应商需经采购比价，单笔金额超过 5 万元的须三家比价。"
     "供应商资质每年复审一次，复审不通过的暂停合作。所有合同须经天元法务审核。"),
    ("policy-data", "数据分级",
     "数据分为公开、内部、机密、绝密四级。绝密数据仅限授权人员访问，"
     "且访问行为全量留痕。内部数据可在公司网络内共享，不得外发。"),
]

RUNBOOKS = [
    ("runbook-oncall", "值班与故障响应",
     "技术部实行 7×24 轮值，每周一轮换。值班人须在 15 分钟内响应 P0 告警、"
     "1 小时内响应 P1 告警。故障升级路径为：值班人 → 组长 → 李伟。"),
    ("runbook-release", "发布流程",
     "发布窗口为每周二、周四 20:00 至 22:00。发布前须完成回归测试并在群内公告。"
     "任何发布必须具备可回滚方案，回滚操作须在 10 分钟内可完成。"),
    ("runbook-backup", "备份与恢复",
     "核心数据库每日全量备份一次，每 5 分钟增量备份一次。"
     "备份保留 30 天。每季度执行一次恢复演练，演练结果须归档。"),
    ("runbook-capacity", "容量规划",
     "容量评估每月执行一次，以峰值 QPS 的 1.5 倍作为扩容触发线。"
     "报表服务单实例上限 1200 QPS，超过后由调度层自动横向扩容。"),
    ("runbook-access", "权限申请",
     "生产环境访问权限须由组长申请、李伟审批，有效期最长 90 天。"
     "数据库写权限仅限值班人在故障处理期间临时开通，处理完毕后当日回收。"),
    ("runbook-monitor", "监控与告警",
     "核心指标包括接口成功率、P99 延迟、错误率、队列积压。"
     "告警分为 P0/P1/P2 三级，P0 直接电话通知值班人，P1 推送企业微信。"),
]


def gen_policies() -> list[tuple[str, str]]:
    return [(f"{slug}.md", f"# {title}\n\n{body}\n") for slug, title, body in POLICIES]


def gen_runbooks() -> list[tuple[str, str]]:
    out = []
    for slug, title, body in RUNBOOKS:
        out.append((f"{slug}.md", f"# {title}\n\n{body}\n"))
    return out


def gen_acronyms() -> list[tuple[str, str]]:
    """每个缩写单独一篇，标题里带缩写本身——这是最典型的盲区词。"""
    out = []
    for abbr, full, body in ACRONYMS:
        content = (
            f"# {abbr}（{full}）\n\n"
            f"{body}。\n\n"
            f"本指标由技术部与相关业务部门共同维护，口径变更需经架构治理周会确认。\n"
        )
        out.append((f"metric-{abbr.lower()}.md", content))
    return out


def gen_products() -> list[tuple[str, str]]:
    out = []
    for model, desc, version in PRODUCTS:
        content = (
            f"# {model}\n\n"
            f"{model} 是{desc}，当前发布版本为 {version}。\n\n"
            f"{model} 由技术部负责维护，产品侧对接人为王芳。\n"
            f"版本 {version} 的变更记录与已知问题登记在内部知识库，升级前须查阅。\n"
        )
        out.append((f"product-{model.lower()}.md", content))
    return out


def gen_projects() -> list[tuple[str, str]]:
    out = []
    for code, name, team, lead, scope, period in PROJECTS:
        content = (
            f"# {name}（{code}）\n\n"
            f"本项目编号 {code}，由{team}承担，负责人是{lead}。\n\n"
            f"项目范围：{scope}。项目周期为 {period}。\n"
            f"立项审批已完成，预算与里程碑登记在项目管理系统，变更须走变更评审。\n"
        )
        out.append((f"project-{code.lower()}.md", content))
    return out


def gen_tickets() -> list[tuple[str, str]]:
    out = []
    for code, title, product, level, owner in TICKETS:
        content = (
            f"# 故障工单 {code}：{title}\n\n"
            f"工单编号 {code}，故障等级 {level}，影响产品 {product}，处理人{owner}。\n\n"
            f"现象：{title}。已按值班与故障响应流程完成定位与修复，复盘结论归档。\n"
        )
        out.append((f"ticket-{code.lower()}.md", content))
    return out


def gen_meetings() -> list[tuple[str, str]]:
    out = []
    for code, date, title, chair, summary in MEETINGS:
        content = (
            f"# {title}纪要（{code}）\n\n"
            f"会议编号 {code}，时间 {date}，主持人{chair}。\n\n"
            f"决议：{summary}。\n"
            f"本纪要经与会人确认，决议自发布之日起生效。\n"
        )
        out.append((f"meeting-{code.lower()}.md", content))
    return out


def gen_externals() -> list[tuple[str, str]]:
    """罕见人名一律给**编制外**角色，避免改变部门人数（32/14/18/9/11 = 84）。"""
    roles = [
        ("供应商对接人", "负责合同履约与交付验收对接"),
        ("外部顾问", "按项目提供专项咨询，不参与日常管理"),
        ("外部审计联系人", "年度审计期间的资料对接窗口"),
        ("法务顾问", "合同审核与合规咨询"),
        ("设计外包对接人", "视觉物料交付与验收"),
        ("物流服务对接人", "样机与物料寄送协调"),
        ("云服务客户经理", "资源采购与技术支持协调"),
        ("背景调查服务对接人", "候选人背景调查流程对接"),
    ]
    out = []
    for (name, company), (role, duty) in zip(RARE_NAMES, roles):
        content = (
            f"# 外部联系人：{name}（{company}）\n\n"
            f"{name}是{company}的{role}，{duty}。\n\n"
            f"{name}不属于公司编制，不计入部门人数。"
            f"日常沟通由对应业务部门负责人对接，合同事宜统一归口财务部。\n"
        )
        out.append((f"external-{company}.md", content))
    return out


def build_blindspots() -> list[dict]:
    """把植入的盲区词登记下来，附一句"正常人会怎么问它"。

    探针题直接从这里生成——这样"我测的就是我植入的"这件事是显式的，
    不会出现"随手写了些文档然后随手编了几道题"那种循环论证。
    """
    spots = []
    for code, name, _team, _lead, _scope, _period in PROJECTS:
        spots.append({"term": code, "kind": "项目编号",
                      "file": f"project-{code.lower()}.md",
                      "question": f"{code} 是哪个项目？由谁负责？"})
    for code, title, _p, _l, _o in TICKETS:
        spots.append({"term": code, "kind": "工单编号",
                      "file": f"ticket-{code.lower()}.md",
                      "question": f"工单 {code} 是什么问题？"})
    for code, _date, title, _chair, _s in MEETINGS:
        spots.append({"term": code, "kind": "会议编号",
                      "file": f"meeting-{code.lower()}.md",
                      "question": f"{code} 这次会议的决议是什么？"})
    for model, _desc, version in PRODUCTS:
        spots.append({"term": model, "kind": "产品型号",
                      "file": f"product-{model.lower()}.md",
                      "question": f"{model} 是什么产品？当前版本是多少？"})
    for abbr, full, _body in ACRONYMS:
        spots.append({"term": abbr, "kind": "英文缩写",
                      "file": f"metric-{abbr.lower()}.md",
                      "question": f"{abbr} 的口径是怎么规定的？"})
    for (name, company), _ in zip(RARE_NAMES, range(len(RARE_NAMES))):
        spots.append({"term": name, "kind": "罕见人名",
                      "file": f"external-{company}.md",
                      "question": f"{name}是谁？"})
    return spots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="先删掉上次生成的文件")
    args = ap.parse_args()

    if args.clean:
        removed = 0
        for path in CORPUS.glob("*.md"):
            if path.name not in ORIGINAL:
                path.unlink()
                removed += 1
        print(f"已删除上次生成的 {removed} 篇")

    files: list[tuple[str, str]] = []
    for group in (gen_policies(), gen_runbooks(), gen_acronyms(), gen_products(),
                  gen_projects(), gen_tickets(), gen_meetings(), gen_externals()):
        files.extend(group)

    clash = [name for name, _ in files if name in ORIGINAL]
    if clash:
        print(f"生成的文件名和原始语料冲突：{clash}", file=sys.stderr)
        return 1

    for name, content in files:
        (CORPUS / name).write_text(content, encoding="utf-8")

    spots = build_blindspots()
    BLINDSPOTS.write_text(json.dumps(spots, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    # 探针题直接由盲区词表生成：「我测的就是我植入的」这件事必须是显式的，
    # 否则就成了"随手写文档 + 随手编题目"的循环论证。
    probes = [
        {
            "id": f"bs-{i:02d}",
            "category": f"专有名词·{spot['kind']}",
            "question": spot["question"],
            "required": [spot["file"]],
            "why": f"植入的{spot['kind']}「{spot['term']}」。"
                   f"这类词转成向量后语义信息很弱，是向量检索的天然盲区",
        }
        for i, spot in enumerate(spots, start=1)
    ]
    (ROOT / "eval" / "probes_blindspot.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"探针题 {len(probes)} 道 → eval/probes_blindspot.json")

    total = sorted(CORPUS.glob("*.md"))
    chars = sum(p.stat().st_size for p in total)
    print(f"生成 {len(files)} 篇，语料共 {len(total)} 篇 / 约 {chars} 字节")
    print(f"盲区词 {len(spots)} 个 → {BLINDSPOTS.relative_to(ROOT)}")
    kinds: dict[str, int] = {}
    for s in spots:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    for k, v in kinds.items():
        print(f"  {k:<8}{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
