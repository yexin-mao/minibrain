"""生成长文档语料。

## 为什么单独一个脚本

`gen_corpus.py` 生成的是短文档（中位数 91 字符），**84 篇切出 84 个片段——
每篇正好一片，切分逻辑一次都没被触发过**。这意味着：

    CHUNK_SIZE / CHUNK_OVERLAP 两个配置项目前完全是摆设
    「切分边界把语义切断」测不了
    「overlap 有没有用」测不了
    「父子索引」测不了
    rerank 也测不出效果（候选集太小）

真实企业文档是几千字的方案、报告、手册。这个脚本补上那一类。

## 设计要点（不是随便凑字数）

**① 长度要跨过 chunk_size**
   每篇 3000~8000 字符，`chunk_size=800` 下切成 5~12 片。

**② 关键事实要散落在不同小节**
   这样才能测出"答案跨片段"的情况——同一篇文档里，A 事实在第 2 片、
   B 事实在第 7 片，检索只召回一片就答不全。

**③ 要有跨文档引用**
   长文档之间互相引用项目编号、工单号、指标名，
   延续 gen_corpus.py 建立的实体网络，不另起一套。

**④ 一致性约束照旧**
   部门人数 32/14/18/9/11 = 84 不能被改变；已有人员只能引用不能改归属。

## 已知取舍

内容是模板拼装的，不如真实文档自然。但评测语料要的是**可控**——
每个事实在哪一篇、哪一节、离边界多远，都必须是已知的，
否则测出来的差异说不清是切分的功劳还是语料的偶然。

跑法：uv run --no-sync python scripts/gen_long_docs.py
      uv run --no-sync python scripts/gen_long_docs.py --clean
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
MANIFEST = ROOT / "eval" / "long_docs_manifest.json"

PREFIX = "long-"          # 生成的文件一律以此开头，--clean 只删这些

TEAMS = [
    ("后端组", "张敏", "技术部", 8, "核心交易服务"),
    ("前端组", "刘洋", "技术部", 7, "Web 与移动端界面"),
    ("算法组", "陈刚", "技术部", 12, "推荐系统和风控模型"),
    ("产品设计组", "胡静", "产品部", 6, "交互设计与视觉规范"),
    ("用户研究组", "吴倩", "产品部", 5, "用户访谈与可用性测试"),
]

PROJECTS = [
    ("PRJ-2026-0142", "Beta", "算法组", "陈刚", "推荐策略重构"),
    ("PRJ-2026-0155", "Gamma", "前端组", "刘洋", "管理后台改版"),
    ("PRJ-2026-0163", "Delta", "后端组", "张敏", "交易链路拆分"),
    ("PRJ-2026-0178", "Epsilon", "用户研究组", "吴倩", "企业客户访谈计划"),
    ("PRJ-2026-0191", "Zeta", "品牌组", "冯磊", "品牌视觉升级"),
    ("PRJ-2026-0204", "Eta", "会计组", "曹敏", "费用系统对接"),
    ("PRJ-2026-0217", "Theta", "产品设计组", "胡静", "设计规范统一"),
    ("PRJ-2026-0223", "Iota", "招聘组", "何静", "校园招聘扩编"),
]

PRODUCTS = [
    ("X7-Pro", "企业版核心引擎", "v2.3.1"),
    ("X7-Lite", "企业版轻量引擎", "v2.3.1"),
    ("S3-Max", "标准版数据采集器", "v1.8.4"),
    ("S3-Mini", "标准版边缘节点", "v1.8.4"),
    ("K2-Air", "移动端 SDK", "v0.9.12"),
    ("R5-Nova", "报表渲染服务", "v4.1.0"),
]

FILLER = [
    "本节内容经相关责任人确认，如与后续发布的正式文件冲突，以正式文件为准。",
    "实施过程中如遇边界情况，由对应负责人判断并在周会同步，不单独发文。",
    "涉及跨部门协作的事项，由发起方提前三个工作日知会相关方。",
    "以上安排自发布之日起执行，历史遗留事项按原约定处理至自然结束。",
    "相关记录统一归档到内部知识库对应目录，保留期与公司文档管理规定一致。",
    "执行过程中产生的例外，需在下一次评审会上说明原因并记录在案。",
    "本条款的解释权归对应归口部门，如有歧义以中文表述为准。",
    "所涉数据的访问遵循数据分级规定，未获授权不得导出到外部环境。",
]


PAD_SCALE = 5          # 填充倍数。真实技术方案/季度报告是几千字，不是几百字


def _pad(rng: random.Random, sentences: int) -> str:
    """填充段落。让文档达到能触发切分的长度，同时不引入新的可检索事实。

    ★ 刻意用无信息量的套话：如果填充里含有真事实，
    评测就分不清"检索命中"是因为切分好还是因为填充里碰巧有答案。
    """
    return "".join(rng.choice(FILLER) for _ in range(sentences * PAD_SCALE))


def _design_doc(rng: random.Random, code, name, team, lead, scope) -> tuple[str, list[dict]]:
    """技术方案文档。关键事实刻意散落在不同小节，用来测「答案跨片段」。"""
    facts = []
    body = [f"# {name} 项目技术方案（{code}）\n"]

    body.append(f"\n## 1. 背景与目标\n\n{name} 项目编号 {code}，由{team}承担，"
                f"负责人是{lead}，项目范围为{scope}。\n\n{_pad(rng, 6)}\n")
    facts.append({"section": "1. 背景与目标", "fact": f"{code} 由{team}承担，负责人{lead}"})

    body.append(f"\n## 2. 现状问题\n\n当前实现存在三个已知问题：接口响应在峰值时段抖动明显；"
                f"配置分散在多处导致变更容易遗漏；缺少灰度手段，一旦回滚影响面较大。\n\n{_pad(rng, 8)}\n")

    rto = rng.choice([15, 30, 45, 60])
    body.append(f"\n## 3. 方案设计\n\n整体分三期推进。第一期完成核心链路拆分，"
                f"第二期接入灰度，第三期做数据迁移。\n\n"
                f"本项目的恢复时间目标（RTO）定为 {rto} 分钟，低于该值视为达标。\n\n{_pad(rng, 10)}\n")
    facts.append({"section": "3. 方案设计", "fact": f"{name} 项目的 RTO 定为 {rto} 分钟"})

    qps = rng.choice([600, 900, 1200, 1500])
    body.append(f"\n## 4. 容量与性能\n\n压测结论：单实例峰值可承载 {qps} QPS，"
                f"超过该值由调度层横向扩容。\n\n{_pad(rng, 8)}\n")
    facts.append({"section": "4. 容量与性能", "fact": f"{name} 项目单实例峰值 {qps} QPS"})

    body.append(f"\n## 5. 风险与依赖\n\n本方案依赖配置中心与灰度平台，"
                f"两者的可用性直接影响上线节奏。\n\n{_pad(rng, 8)}\n")

    reviewer = rng.choice(["李伟", "王芳", "周涛"])
    body.append(f"\n## 6. 评审结论\n\n方案已通过评审，评审人{reviewer}。"
                f"要求第一期上线前补齐回滚预案。\n\n{_pad(rng, 6)}\n")
    facts.append({"section": "6. 评审结论", "fact": f"{name} 项目方案的评审人是{reviewer}"})

    return "".join(body), facts


def _quarterly_report(rng: random.Random, team, lead, dept, size, duty) -> tuple[str, list[dict]]:
    """季度报告。事实分布在开头和结尾，中间大段填充——专门测跨片段召回。"""
    facts = []
    body = [f"# {team}季度工作报告\n"]

    body.append(f"\n## 一、团队概况\n\n{team}隶属{dept}，组长是{lead}，现有 {size} 人，"
                f"负责{duty}。\n\n{_pad(rng, 8)}\n")
    facts.append({"section": "一、团队概况", "fact": f"{team}组长{lead}，{size} 人"})

    body.append(f"\n## 二、本季度进展\n\n按季度计划推进各项工作，整体达成预期。"
                f"其中重点事项的推进节奏与年初规划基本一致。\n\n{_pad(rng, 12)}\n")

    body.append(f"\n## 三、遇到的问题\n\n跨团队协作的对齐成本高于预期，"
                f"部分依赖方的交付时间存在不确定性。\n\n{_pad(rng, 10)}\n")

    onboard = rng.choice([1, 2, 3])
    body.append(f"\n## 四、人员变动\n\n本季度{team}新增 {onboard} 名成员，无人员离开。"
                f"新成员均已完成入职培训。\n\n{_pad(rng, 6)}\n")
    facts.append({"section": "四、人员变动", "fact": f"{team}本季度新增 {onboard} 人"})

    body.append(f"\n## 五、下季度计划\n\n继续推进既定事项，同时补齐本季度识别出的短板。"
                f"具体排期在季度启动会上确认。\n\n{_pad(rng, 8)}\n")
    return "".join(body), facts


def _product_manual(rng: random.Random, model, desc, version) -> tuple[str, list[dict]]:
    """产品手册。版本号和配置项散落各处，用来测标识符在长文档里的检索。"""
    facts = []
    body = [f"# {model} 产品手册\n"]

    body.append(f"\n## 1. 产品简介\n\n{model} 是{desc}，当前发布版本 {version}。"
                f"由技术部维护，产品侧对接人为王芳。\n\n{_pad(rng, 6)}\n")
    facts.append({"section": "1. 产品简介", "fact": f"{model} 当前版本 {version}"})

    body.append(f"\n## 2. 部署要求\n\n最低配置为 4 核 8G，推荐 8 核 16G。"
                f"依赖的中间件版本需与发布说明保持一致。\n\n{_pad(rng, 8)}\n")

    port = rng.choice([8080, 8443, 9090, 9443])
    body.append(f"\n## 3. 配置说明\n\n{model} 默认监听端口为 {port}，"
                f"可通过配置文件覆盖。生产环境须显式指定，不使用默认值。\n\n{_pad(rng, 10)}\n")
    facts.append({"section": "3. 配置说明", "fact": f"{model} 默认端口 {port}"})

    body.append(f"\n## 4. 常见问题\n\n启动失败多数由端口占用或配置文件路径错误导致，"
                f"排查时优先确认这两项。\n\n{_pad(rng, 10)}\n")

    body.append(f"\n## 5. 升级说明\n\n升级到 {version} 前须查阅变更记录与已知问题，"
                f"跨大版本升级需先在预发环境验证。\n\n{_pad(rng, 8)}\n")
    return "".join(body), facts


TICKETS = [
    ("TICKET-88231", "报表导出超时", "R5-Quasar", "P1", "张敏"),
    ("TICKET-88407", "移动端崩溃率上升", "K2-Air", "P0", "刘洋"),
    ("TICKET-88512", "推荐结果重复", "X7-Pro", "P1", "陈刚"),
    ("TICKET-88690", "边缘节点心跳丢失", "S3-Mini", "P1", "张敏"),
    ("TICKET-88745", "报销单据无法上传", "R5-Nova", "P1", "曹敏"),
    ("TICKET-88823", "数据采集延迟", "S3-Max", "P0", "陈刚"),
]

HANDBOOKS = [
    ("差旅与报销", "报销", "发票需在消费后 30 天内提交"),
    ("考勤与假期", "考勤", "核心工作时间为 10:00 至 16:00"),
    ("绩效与晋升", "绩效", "绩效考核每半年一次，分 S/A/B/C 四档"),
    ("信息安全", "安全", "机密文件不得存放在个人设备或公有云盘"),
    ("供应商合作", "采购", "单笔金额超过 5 万元的须三家比价"),
    ("研发流程", "研发", "发布窗口为每周二、周四 20:00 至 22:00"),
]


def _postmortem(rng: random.Random, code, title, product, level, owner) -> tuple[str, list[dict]]:
    """故障复盘。时间线、根因、改进项分布在不同小节。"""
    facts = []
    body = [f"# 故障复盘报告 {code}\n"]
    minutes = rng.choice([12, 27, 43, 68, 95])
    body.append(f"\n## 1. 事件概述\n\n工单 {code}，故障等级 {level}，影响产品 {product}，"
                f"处理人{owner}。现象为{title}。本次故障持续 {minutes} 分钟。\n\n{_pad(rng, 8)}\n")
    facts.append({"section": "1. 事件概述", "fact": f"{code} 持续 {minutes} 分钟"})

    body.append(f"\n## 2. 时间线\n\n告警触发后值班人按流程响应，定位阶段耗时最长，"
                f"修复与验证在同一窗口内完成。\n\n{_pad(rng, 10)}\n")

    cause = rng.choice(["配置未同步", "内存泄漏", "连接池耗尽", "上游限流", "索引缺失"])
    body.append(f"\n## 3. 根因分析\n\n经排查，本次故障的直接根因是{cause}。"
                f"该问题在预发环境未复现，因为流量特征不同。\n\n{_pad(rng, 10)}\n")
    facts.append({"section": "3. 根因分析", "fact": f"{code} 的根因是{cause}"})

    body.append(f"\n## 4. 改进项\n\n共产出三项改进：补充监控指标、增加预发压测、"
                f"完善回滚预案。均已登记并指定责任人。\n\n{_pad(rng, 10)}\n")

    body.append(f"\n## 5. 复盘结论\n\n本次响应符合值班流程要求，改进项须在下个迭代内闭环。"
                f"\n\n{_pad(rng, 6)}\n")
    return "".join(body), facts


def _handbook(rng: random.Random, title, kind, rule) -> tuple[str, list[dict]]:
    """制度手册详细版。条款分散在多章，用来测「同一制度的不同条款跨片段」。"""
    facts = []
    body = [f"# {title}手册\n"]
    body.append(f"\n## 第一章 适用范围\n\n本手册适用于公司全体正式员工，"
                f"试用期员工除特别说明外同样适用。\n\n{_pad(rng, 10)}\n")

    body.append(f"\n## 第二章 基本规定\n\n{rule}。该规定为{kind}类事项的基础要求，"
                f"各部门不得自行放宽。\n\n{_pad(rng, 12)}\n")
    facts.append({"section": "第二章 基本规定", "fact": rule})

    days = rng.choice([3, 5, 7, 10, 15])
    body.append(f"\n## 第三章 申请流程\n\n需提前 {days} 个工作日提交申请，"
                f"经直属上级审批后生效。紧急情况可事后补办，但须说明原因。\n\n{_pad(rng, 12)}\n")
    facts.append({"section": "第三章 申请流程", "fact": f"{title}需提前 {days} 个工作日申请"})

    body.append(f"\n## 第四章 违规处理\n\n首次违规由直属上级提醒；再次违规计入当期绩效；"
                f"情节严重的按公司相关规定处理。\n\n{_pad(rng, 10)}\n")

    owner = rng.choice(["孙红", "周涛", "李伟", "王芳"])
    body.append(f"\n## 第五章 解释与修订\n\n本手册由{owner}归口维护，"
                f"修订须经管理层评审。\n\n{_pad(rng, 8)}\n")
    facts.append({"section": "第五章 解释与修订", "fact": f"{title}手册由{owner}归口维护"})
    return "".join(body), facts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--seed", type=int, default=20260805, help="固定种子，保证可复现")
    args = ap.parse_args()

    if args.clean:
        removed = sum(1 for p in CORPUS.glob(f"{PREFIX}*.md") if p.unlink() is None)
        print(f"已删除上次生成的 {removed} 篇长文档")

    rng = random.Random(args.seed)
    generated: list[tuple[str, str, list[dict]]] = []

    for code, name, team, lead, scope in PROJECTS:
        text, facts = _design_doc(rng, code, name, team, lead, scope)
        generated.append((f"{PREFIX}design-{code.lower()}.md", text, facts))

    for team, lead, dept, size, duty in TEAMS:
        for quarter in ("q1", "q2", "q3", "q4"):
            text, facts = _quarterly_report(rng, team, lead, dept, size, duty)
            generated.append((f"{PREFIX}report-{quarter}-{lead}.md", text, facts))

    for model, desc, version in PRODUCTS:
        text, facts = _product_manual(rng, model, desc, version)
        generated.append((f"{PREFIX}manual-{model.lower()}.md", text, facts))

    for code, title, product, level, owner in TICKETS:
        text, facts = _postmortem(rng, code, title, product, level, owner)
        generated.append((f"{PREFIX}postmortem-{code.lower()}.md", text, facts))

    for title, kind, rule in HANDBOOKS:
        text, facts = _handbook(rng, title, kind, rule)
        generated.append((f"{PREFIX}handbook-{kind}.md", text, facts))

    for name, text, _ in generated:
        (CORPUS / name).write_text(text, encoding="utf-8")

    MANIFEST.write_text(json.dumps([
        {"file": name, "chars": len(text), "facts": facts}
        for name, text, facts in generated
    ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 报告切分效果——这才是这次的目的
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from minibrain.config import get_config
    from minibrain.modules.vector_rag.chunking import split_text

    cfg = get_config()
    all_docs = sorted(CORPUS.glob("*.md"))
    chunk_counts = [len(split_text(p.read_text(encoding="utf-8"),
                                   cfg.chunk_size, cfg.chunk_overlap)) for p in all_docs]
    lengths = [len(p.read_text(encoding="utf-8")) for p in all_docs]

    print(f"\n生成 {len(generated)} 篇长文档")
    print(f"语料共 {len(all_docs)} 篇 / {sum(lengths):,} 字符")
    print(f"chunk_size={cfg.chunk_size} overlap={cfg.chunk_overlap} 下：")
    print(f"  切出片段总数：{sum(chunk_counts)}   （扩之前是 84，每篇 1 片）")
    print(f"  切成 2 片以上的文档：{sum(1 for c in chunk_counts if c > 1)} 篇 ← 切分终于被触发了")
    print(f"  单篇最多切成：{max(chunk_counts)} 片")
    print(f"  长文档平均：{sum(len(t) for _, t, _ in generated) / len(generated):,.0f} 字符")
    print(f"\n事实清单已写入 {MANIFEST.relative_to(ROOT)}"
          f"（{sum(len(f) for _, _, f in generated)} 条，标注了每个事实在哪一节）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
