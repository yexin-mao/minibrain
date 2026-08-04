# minibrain

最小知识中台。**两条范式不同的检索链路，各自独立存储和判权限；一个统一入口按问题类型编排。**

不是"又一个 RAG demo"——一条链路的 RAG 没有边界需要验证。这个项目要证明的是边界本身。

```
用户提问
  └─ Agent（手写 tool loop，两个工具）
       ├─ vector-rag  文档 → 切分 → embedding + BM25 → RRF 融合  "报销标准是怎么规定的？"
       └─ table-rag   CSV  → 物理表 → 受限只读 SQL            "华东区销售额合计是多少？"
```

第二条链路是表格而不是图谱，是刻意的：**一张报表切碎再向量召回回来，算不出正确的合计。**
这是"两条链路不能合并"最无可辩驳的证据，而且它不需要 LLM 抽取、不需要图数据库、不需要第二种语言。

> 第一次读这个仓库，先跑这个——它把两条链路每一步的中间产物原样打印出来，
> 包括"把表格误传进文档链路会被切成什么样"：
>
> ```sh
> uv run --no-sync python scripts/explain.py
> ```

## 跑起来

前置：PostgreSQL（跑着就行，不需要任何扩展）、[uv](https://docs.astral.sh/uv/)。

```sh
uv sync
cp .env.example .env          # 填 DATABASE_URL；两个 API key 先不填也能跑通表格链路
uv run minibrain-init-db      # 建库 + 跑 schema.sql，幂等
uv run minibrain-create-user alice alice123   # 第一个用户自动是管理员
uv run uvicorn minibrain.web.app:app --reload --port 8000
```

打开 http://127.0.0.1:8000 。上传 `.csv` 走表格链路，`.md` / `.txt` 走文档链路。

不填 API key 时：表格链路完整可用（不需要模型），文档链路的文档会落 `failed` 并写明原因，
提问会提示 `agent_not_configured`。**这是设计好的降级，不是坏掉了。**

## 测试

```sh
uv run pytest            # 50 个用例，不需要 API key 也能跑
uv run pytest -k 权限     # 或 guardrail / cleanup
```

**不 mock 数据库**——要验证的恰恰是 SQL 里的权限过滤和只读事务，
mock 掉数据库等于把被测对象本身删了。所以测试建的是真实用户和真实数据，
session fixture 结束时统一清理，`test_cleanup.py` 专门验证清理本身。

配了 embedding key 跑 49 passed / 1 skipped，没配跑 48 / 2——
**两条分支都被覆盖**：没配 key 时文档必须落 `failed` 并写明原因，不许假装 ready。
CI 刻意不注入密钥，所以那条降级路径由 CI 守门（本地有 key 反而测不到）。

测试有没有牙，是验证过的：把 `_visibility_clause()` 的非管理员分支改成永真
（模拟"漏一个分支"这类典型越权 bug），3 个权限测试立刻失败。

如果异常中断留下了残渣：

```sh
uv run minibrain-purge --list     # 先看会删什么
uv run minibrain-purge            # 清掉 smoke_ / live_ 前缀的用户及其全部数据
```

## 开发流程：改动必须走 PR

```sh
git config core.hooksPath .githooks    # 一次性，启用 pre-push 拦截

git switch -c fix/你的改动
git push -u origin fix/你的改动
gh pr create --fill
gh pr checks --watch                   # 等 CI 绿
gh pr merge --squash --delete-branch
```

**为什么不直接推 main**——这不是形式主义，是实测出来的：

这个仓库的代码几乎全部由 AI 生成。推上 GitHub 之前，本地 87 个测试全绿。
**CI 第一次真跑，连续失败两次，各抓到一个本地永远碰不到的问题：**

| # | 问题 | 为什么本地发现不了 |
|---|---|---|
| 1 | `url.replace(f"/{dbname}", "/postgres")` 把连接串改烂 | `str.replace` 替换所有匹配。CI 的连接串里**用户名恰好等于库名**，本地的不是 |
| 2 | 三个测试在无 API key 时断言恒真 | CI 不带 key → 文档落 `failed` → 清单为空 → `assert X not in 空` 恒真。**其中一个是权限测试，等于从来没测过** |

第二个尤其危险：**假绿比红更糟**——红了你会去修，假绿你以为有覆盖，其实没有。

所以规则是：**本地绿不等于对，CI 绿了才能合。**
CI 的价值不是重复跑本地已经绿的测试，是跑本地跑不到的那条路径。

> 服务端分支保护对免费私有仓库不开放，所以用 `.githooks/pre-push` 客户端兜底。
> 它挡不住 `--no-verify`，但挡得住手滑——而手滑正是实际会发生的那种。

## 评测（结论是测出来的，不是声称的）

这个项目的每个主张都要有数字撑着。评测分两块，结果都在版本库里：

```sh
uv run --no-sync python scripts/probe_vector_rag.py    # 向量链路在什么问题上失效
uv run --no-sync python scripts/probe_multihop.py      # 多跳失败是召回率还是召回时机
uv run --no-sync python scripts/probe_silent_error.py  # 召回不全时模型会不会静默答错
uv run --no-sync python scripts/eval_routing.py        # 43 题路由准确率
```

| 报告 | 结论摘要 |
|---|---|
| [`eval/RESULTS.md`](eval/RESULTS.md) | **混合检索：全局 MRR 0.855 → 0.950，原有题目零损伤**。过程翻了两次车，翻车比结果值钱：① 朴素 RRF 把普通问题的 Recall@5 从 0.803 打到 0.610，根因是中文二元组的「偶然稀有」（`天年` 是「三年年假」切出的伪词）——所以关键词路只负责标识符；② 项目编号纹丝不动，是**结构性平局**（两边名次都是 {1,2}，调 k 也没用），改破平规则后 0.750 → 1.000。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（扩语料） | **扩语料到 84 篇后，同样 10 道题指标掉了约 10 个点**（Complete Recall@10 从 1.000 掉到 0.800）——玩具规模语料上的检索指标不能外推。另测出真正的盲区不是「编号」而是**「前缀相同、只差几位数字的编号」**：问 `MTG-20260617-02`，前三名全是别的会议纪要，正确答案排第 8。而且问题在**排序不在召回**（Recall@5=1.000，MRR 只有 0.489）——这正是混合检索的确切用武之地。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（初版） | **标准 IR 指标全都很漂亮**（MRR=1.000、Hit@k=1.000、Recall@5=0.930、NDCG@5=0.945），但同一份数据换成 Complete Recall@5 只有 0.800，全局聚合类 k=3 时 **0/3**——**标准指标在"必须召回全"的场景下会系统性高估**。多跳失败的根因是"召回时机"不是"召回率"：同样 k=3，单次检索漏掉目标文档（实测排第 6），两步检索命中。 |
| [`eval/ROUTING.md`](eval/ROUTING.md) | 严格路由准确率 **83.7% → 95.3%**（各跑 3 轮）。失败原本集中在"文档·组织事实" 1/6 且三轮一致，根因是 system prompt 只注入了表结构没注入文档清单；改后该类 **6/6**，且三轮零波动、过度调用率反而从 9.3% 降到 4.7%。过程中暴露并修好了一处回归（`tbl-18`）：schema 给了列名没给取值，模型只能猜枚举值。 |
| [`SCALING.md`](SCALING.md) | **这套做法在真实规模下会怎样**：文档清单那段是 `O(文档数)`，1000 万篇文档时不可行。消融数据显示贡献更大的权威来源规则是 `O(1)`——需要替换的只是清单。业界替代方案（语义路由 / 分层索引 / 元数据过滤 / 语义层）与不做的理由（15 篇语料上 MRR 已是 1.000，测不出差别）。 |
| [`eval/PROMPT_ABLATION.md`](eval/PROMPT_ABLATION.md) | 消融实验（43 题 × 4 版本 × 3 轮）验证根因：目标类别 39% → 61% → 78% → **100%** 单调改善。结论是**给资料 ≠ 给判断依据**——光列出文档修不好 `doc-08`，必须明确"冲突时听谁的"。 |

两条最该被记住的：

**1. 向量链路会静默答错。** 问"公司总共多少人"，top_k=5 漏掉财务部，模型答"合计约 73 人"
（真值 84，少算 13%），而回答列了明细、标了来源、语气自然，用户无从察觉。
**检索的缺陷被生成层完美掩盖了**——这是"必须有评测体系"最直接的论据。

**2. 路由错了但答案对了，比路由错更危险。** 三轮答案正确率 100%，路由准确率只有 83.7%，
差额全是"走错链路但因数据重叠碰巧答对"。`doc-08` 是运气用完的地方：走表格链路查出
4 个小组（多了一个记账行），文档里写的是 3 个——**而子串匹配的答案检查没抓到**。
子串匹配只能抓"少答"，抓不到"多答"。

### embedding 维度

`qwen/qwen3-embedding-8b` 实际输出 **4096** 维。代码按 `EMBEDDING_DIMENSIONS` 做
MRL 截断 + 重新归一化（默认降到 1024，与 ff-companybrain 同口径），省 4 倍内存和存储。

query 和 chunk 走同一个函数、同一套截断口径——这两边一旦不同，检索会悄悄变差且极难查。
改 `EMBEDDING_DIMENSIONS` 必须重新索引全部文档。

## 结构

```
schema.sql                  三个 schema。改结构就改这个文件，不要迁移框架
eval/
  corpus/                   15 篇虚构公司文档（非结构化）
  corpus_table/             3 张 CSV（花名册/销售/报销），和文档是同一家公司
  probes.json               10 道检索探针，每题标注"答对必须召回哪几篇"
  routing.json              43 道路由用例，每题标注"该调哪几个工具"
  RESULTS.md / ROUTING.md   ★ 评测结论，全部可复现
tests/                      50 个 pytest 用例（权限/护栏/状态机/清理）
scripts/
  explain.py              ★ 跑一遍就看懂两条链路的差别（中间产物全打印）
  probe_*.py              三个检索探针
  eval_routing.py         43 题路由评测
src/minibrain/
  config.py                 环境变量，一次读取一次校验
  contracts.py              UserContext / ModuleId / Evidence，薄契约
  db.py                     三个连接池，各自锁死 search_path ← 边界的物理落点
  gateway.py                模块分发。唯一允许 import modules/ 的地方
  identity/                 用户名密码 + bcrypt + 会话
  modules/
    vector_rag/             切分 + embedding + 内存余弦
    table_rag/              CSV 建表 + 只读 SQL + 四层护栏
  agent/                    手写 tool loop，不上框架
  web/                      FastAPI + Jinja2 + htmx，无构建步骤
```

应用代码约 2200 行（含 schema 与模板），测试 545 行，评测 631 行。
作为对照，同一个立意的"完整版"是 6.3 万行。

## 四条不将就的规矩

尺寸可以小，这四条不能松——它们事后返工的成本极高，而现在遵守它们的成本几乎为零。

**1. 权限过滤写在 SQL 的 WHERE 里，不是查完再筛。**
`_visibility_clause()` 是两个模块各自唯一的可见性判定，永远出现在 WHERE 里。
`chunks.source_id` 是从 `documents` 冗余下来的，就是为了让过滤不必 join。
应用层后过滤是最容易长出越权 bug 的地方：漏一个分支就是数据泄露。

**2. `core.py` 里永远不 import fastapi。**
模块只收 `UserContext` 和普通参数，只返回 dataclass。
做到这点，后面加 MCP、加 CLI、加定时任务都是白送的。

**3. 业务逻辑不进 `web/app.py`。**
路由只做解析、鉴权、调 gateway、渲染。一旦开了"就这一处先放这儿"的口子，
它会长成一个上千行、几十个分支的路由文件，然后再也搬不回去。

**4. 状态机带 `failed` 态，失败不伪装成 ready。**
上传接口只登记就立刻返回，处理在后台跑，前端轮询状态。
向量化是分钟级的，同步阻塞在生产上会被反代掐断——这个坑要在第一天就避开。

## 表格链路的四层护栏

LLM 会写 SQL，所以护栏必须是纵深的，任何一层单独都不够：

1. **表名白名单** —— 先用 SQL 算出当前用户看得见哪些表，权限的真正落点
2. **只允许单条 SELECT / WITH**
3. **强制外层 LIMIT**（把语句包进子查询）
4. **只读事务 + statement_timeout** —— 在连接层面，不依赖上面三层的正确性

第 4 层用 `SET default_transaction_read_only = on` 实现，不需要建 PG 角色，
也就不需要超级用户权限。物理表名由模块生成（`t_<8位hex>`），用户输入永远不进标识符。

## 留好的缝

这些都是设计好的升级路径，不是欠的债：

| 现在 | 将来 | 改哪里 |
|---|---|---|
| 单进程、gateway 是 dispatch table | 模块拆成独立 HTTP 服务 | `gateway.call` 改成带 `x-ff-*` header 的 fetch，调用方零改动 |
| 内存 numpy 余弦 | pgvector | `vector_rag/core.py` 的 `search` 一个函数 |
| 一库三 schema | 三个独立数据库 | `db.py` 的 conninfo；每个 schema 一次 pg_dump |
| 手写 tool loop | LangGraph + checkpoint | `agent/loop.py`；等真需要长会话摘要时再换 |
| 两条链路 | 加 GraphRAG 第三条 | 新建 `modules/graph_rag/`，在 `gateway._REGISTRY` 注册 |

加第三条链路是最能验证边界的时刻——前两条已经把 gateway 和 UserContext 的形状压出来了。

## 已知边界

- 表名白名单靠正则提取 `FROM` / `JOIN` 后的标识符。只读事务和 search_path 锁定是兜底，
  但足够刁钻的构造可能绕过白名单这一层。真要上生产，这里应该换成真正的 SQL 解析。
- 会话存明文 token，没有轮换和刷新。
- 全量加载可见 chunk 到内存算余弦，几万条以上会明显变慢。
- 后台处理用 FastAPI `BackgroundTasks`，进程重启会丢在途任务（原文已落库，重跑即可）。
- **路由基线只有 83.7%**，且失败集中在"事实同时存在于文档和表"的题上。
  根因和四条修法都写在 [`eval/ROUTING.md`](eval/ROUTING.md)，还没实施——
  实施后必须重跑评测对比数字，否则"我优化了路由"只是一句话。
- **路由方案不可扩展**：文档清单是 `O(文档数)`，10 万篇以上就超上下文窗口。
  可扩展的替代方案与触发条件见 [`SCALING.md`](SCALING.md)——现在不做是因为 15 篇语料上测不出差别。
- 列取值只对**低基数文本列**注入（≤12 个取值、≤40 字符）。高基数列模型仍然只能猜——
  大规模下正确做法是按需检索（value retrieval），见 [`SCALING.md`](SCALING.md)。
- 单轮问答，没有会话、没有历史、没有上下文压缩。
- 答案检查用子串匹配，只能抓"少答"，抓不到"多答"（`doc-08` 就是漏网的例子）。
